//! *Tools ▶ Transfer by relationship*: place a structure into the other
//! workspace by its spatial relationship to a reference structure.
//!
//! A STAR target defined on one patient's imaging cannot be propagated onto
//! another patient (or another posture) by registration alone when the two
//! workspaces share no anatomy-to-anatomy correspondence for it. What travels
//! instead is the *relationship*: the target's offset from the centroid of a
//! reference structure both workspaces can segment - typically the heart. The
//! target lands in the destination at the same offset from the destination's
//! reference structure, keeping its shape; deformable adaptation, when
//! wanted, is the propagation tool's job afterwards.

use crate::motion;

use super::*;

/// The window's state.
pub(super) struct TransferDialog {
    /// Workspace the target comes from; it lands on the other one.
    pub src_slot: usize,
    /// Candidate index of the target in the source workspace.
    pub target: Option<usize>,
    /// Candidate index of the reference structure in the source workspace.
    pub src_ref: Option<usize>,
    /// Candidate index of the reference structure in the destination.
    pub dst_ref: Option<usize>,
    pub status: Option<String>,
    /// The offset the two reference structures give, as a matrix - what the
    /// window shows in its transform grid before anyone edits it.
    pub auto: Option<crate::registration::Mat4>,
    /// The picks `auto` was worked out for. Two structures have to be
    /// rasterized to find their centroids, which is a thing to do when the
    /// choice changes and not on every frame.
    pub auto_for: (usize, Option<usize>, Option<usize>),
}

impl ViewerApp {
    pub(super) fn open_transfer_dialog(&mut self, src_slot: usize) {
        let mut d = TransferDialog {
            src_slot,
            target: None,
            src_ref: None,
            dst_ref: None,
            status: None,
            auto: None,
            auto_for: (usize::MAX, None, None),
        };
        // Pre-pick reference structures by the obvious name.
        let guess = |cands: &[(super::combine::ItemRef, String)]| {
            cands.iter().position(|(_, l)| {
                let l = l.to_lowercase();
                l.contains("heart") || l.contains("herz")
            })
        };
        d.src_ref = guess(&self.combine_candidates(src_slot));
        d.dst_ref = guess(&self.combine_candidates(1 - src_slot));
        self.transfer_dialog = Some(d);
    }

    /// Carry the target across, synchronously - a placement and one
    /// nearest-neighbour resampling over the target's bounding box.
    fn transfer_now(&mut self) {
        let Some(d) = &self.transfer_dialog else {
            return;
        };
        let (src, dst) = (d.src_slot, 1 - d.src_slot);
        let (sel_target, sel_src_ref, sel_dst_ref) = (d.target, d.src_ref, d.dst_ref);
        let pick = |slot: usize, sel: Option<usize>| {
            sel.and_then(|i| self.combine_candidates(slot).get(i).cloned())
        };
        // A matrix typed into the window says where the target goes; without
        // one it is the relationship between the two reference structures,
        // and only then are they needed at all.
        let manual = self.transfer_matrix.transform(crate::geometry::Vec3::ZERO);
        let Some((it, _)) = pick(src, sel_target) else {
            self.transfer_status("Pick the target structure first.");
            return;
        };
        let Some((tm, tg, tname, tcolor)) = self.item_mask_grid(src, it) else {
            self.transfer_status("The target structure is gone or empty.");
            return;
        };
        let (placement, name, how) = match manual {
            Some(t) => (
                t,
                format!("{tname} (matrix)"),
                "placed by the matrix typed into the window".to_string(),
            ),
            None => {
                let (Some((ir, _)), Some((id_, _))) =
                    (pick(src, sel_src_ref), pick(dst, sel_dst_ref))
                else {
                    self.transfer_status(
                        "Pick both reference structures first, or type a transform matrix \
                         in and tick 'Use this matrix'.",
                    );
                    return;
                };
                let (Some((rm, rg, rname, _)), Some((dm, dg, dname, _))) =
                    (self.item_mask_grid(src, ir), self.item_mask_grid(dst, id_))
                else {
                    self.transfer_status("One of the reference structures is gone or empty.");
                    return;
                };
                let (Some(c_target), Some(c_src), Some(c_dst)) = (
                    motion::centroid_mm(&tm, &tg),
                    motion::centroid_mm(&rm, &rg),
                    motion::centroid_mm(&dm, &dg),
                ) else {
                    self.transfer_status("One of the structures has no voxels.");
                    return;
                };
                let delta = c_dst - c_src;
                (
                    Transform3::from_matrix(
                        crate::registration::Mat4::translation(delta),
                        crate::geometry::Vec3::ZERO,
                    ),
                    format!("{tname} @ {dname}"),
                    format!(
                        "offset from {rname}: RL {:+.1} · AP {:+.1} · SI {:+.1} mm",
                        c_target.x - c_src.x,
                        c_target.y - c_src.y,
                        c_target.z - c_src.z,
                    ),
                )
            }
        };

        // The destination lattice is the displayed volume of the other
        // workspace - that is where a new segmentation is editable.
        let Some(study) = self.slots[dst].study.as_ref() else {
            return;
        };
        let out_grid = study.volume.grid();
        let mask = map_mask(&tm, &tg, &out_grid, &placement);
        if mask.iter().all(|&v| v == 0) {
            let msg = format!(
                "'{tname}' lands outside workspace {}'s displayed volume - nothing to store.",
                SLOT_NAMES[dst]
            );
            self.transfer_status(&msg);
            return;
        }
        let placed_cm3 = motion::volume_cm3(&mask, &out_grid);
        let dims = out_grid.dims;
        self.add_colored_segmentation(dst, name.clone(), tcolor, dims, &mask);
        let msg = format!(
            "'{name}' stored in workspace {} - {how}, {placed_cm3:.2} cm³.",
            SLOT_NAMES[dst]
        );
        self.transfer_status(&msg);
    }

    /// The one line the window reports its last run on.
    fn transfer_status(&mut self, msg: &str) {
        if let Some(d) = &mut self.transfer_dialog {
            d.status = Some(msg.to_string());
        }
    }

    /// The relationship's own answer as a matrix: the shift from the source
    /// reference structure's centroid to the destination reference's.
    ///
    /// This is what the tool would do by itself, so it is what the transform
    /// grid shows until someone takes it over.
    fn relationship_matrix(
        &self,
        src: usize,
        src_ref: Option<usize>,
        dst_ref: Option<usize>,
    ) -> Option<crate::registration::Mat4> {
        let dst = 1 - src;
        let pick = |slot: usize, sel: Option<usize>| {
            sel.and_then(|i| self.combine_candidates(slot).get(i).cloned())
        };
        let (ir, _) = pick(src, src_ref)?;
        let (id_, _) = pick(dst, dst_ref)?;
        let (rm, rg, _, _) = self.item_mask_grid(src, ir)?;
        let (dm, dg, _, _) = self.item_mask_grid(dst, id_)?;
        let c_src = motion::centroid_mm(&rm, &rg)?;
        let c_dst = motion::centroid_mm(&dm, &dg)?;
        Some(crate::registration::Mat4::translation(c_dst - c_src))
    }

    pub(super) fn transfer_window(&mut self, ctx: &egui::Context) {
        let Some(d) = &self.transfer_dialog else {
            return;
        };
        let src = d.src_slot;
        let dst = 1 - src;
        if !self.slots[src].has_volume() || !self.slots[dst].has_volume() {
            self.transfer_dialog = None;
            return;
        }
        let src_cands: Vec<String> = self
            .combine_candidates(src)
            .into_iter()
            .map(|(_, l)| l)
            .collect();
        let dst_cands: Vec<String> = self
            .combine_candidates(dst)
            .into_iter()
            .map(|(_, l)| l)
            .collect();
        // What the relationship itself gives, recomputed only when the
        // reference structures change.
        let key = (src, d.src_ref, d.dst_ref);
        let computed = if d.auto_for == key {
            d.auto
        } else {
            self.relationship_matrix(src, key.1, key.2)
        };
        let mut run = false;
        let mut close = false;
        let mut swap = false;
        let mut open = true;
        let d = self.transfer_dialog.as_mut().expect("checked above");
        d.auto = computed;
        d.auto_for = key;
        let mut manual = self.transfer_matrix;
        detach::tool_window(
            ctx,
            "transfer",
            "◎ Transfer by relationship",
            &mut open,
            detach::WinOpts::default(),
            |ui| {
                ui.label(format!(
                    "Place a structure of workspace {} into workspace {} at the same offset \
                     from a reference structure (e.g. the heart) - the target-reference \
                     relationship travels, not the image registration.",
                    SLOT_NAMES[src], SLOT_NAMES[dst]
                ));
                ui.add_space(4.0);
                let combo = |ui: &mut egui::Ui,
                             label: &str,
                             item: &mut Option<usize>,
                             list: &[String],
                             salt: &str| {
                    ui.horizontal(|ui| {
                        ui.label(label);
                        index_picker(ui, salt, item, list);
                    });
                };
                combo(
                    ui,
                    &format!("Target ({}):", SLOT_NAMES[src]),
                    &mut d.target,
                    &src_cands,
                    "tr_target",
                );
                // With a matrix in charge the relationship is not consulted,
                // so the two references go quiet rather than looking required.
                ui.add_enabled_ui(!manual.use_it, |ui| {
                    combo(
                        ui,
                        &format!("Reference in {}:", SLOT_NAMES[src]),
                        &mut d.src_ref,
                        &src_cands,
                        "tr_src_ref",
                    );
                    combo(
                        ui,
                        &format!("Reference in {}:", SLOT_NAMES[dst]),
                        &mut d.dst_ref,
                        &dst_cands,
                        "tr_dst_ref",
                    );
                });
                if ui
                    .button(format!("Swap direction (to workspace {})", SLOT_NAMES[src]))
                    .clicked()
                {
                    swap = true;
                }
                ui.add_space(4.0);
                matrix_edit::matrix_editor(ui, &mut manual, computed);
                if manual.use_it {
                    ui.weak(
                        "The target is placed through these numbers instead of by its \
                         offset from the reference structures.",
                    );
                }
                ui.add_space(4.0);
                if let Some(status) = &d.status {
                    ui.label(status.clone());
                }
                ui.horizontal(|ui| {
                    if ui.button("▶ Transfer").clicked() {
                        run = true;
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            },
        );
        self.transfer_matrix = manual;
        if swap {
            if let Some(d) = &mut self.transfer_dialog {
                d.src_slot = 1 - d.src_slot;
                d.target = None;
                d.src_ref = None;
                d.dst_ref = None;
                d.status = None;
            }
            if let Some(slot) = self.transfer_dialog.as_ref().map(|d| d.src_slot) {
                let guess = |cands: Vec<(super::combine::ItemRef, String)>| {
                    cands.iter().position(|(_, l)| {
                        let l = l.to_lowercase();
                        l.contains("heart") || l.contains("herz")
                    })
                };
                let s = guess(self.combine_candidates(slot));
                let t = guess(self.combine_candidates(1 - slot));
                if let Some(d) = &mut self.transfer_dialog {
                    d.src_ref = s;
                    d.dst_ref = t;
                }
            }
        }
        if run {
            self.transfer_now();
        }
        if close || !open {
            self.transfer_dialog = None;
        }
    }
}

/// Resample `mask` (on `from`) onto `to`, shifted by `delta` in patient
/// coordinates: `out(p) = mask(p − delta)`. Nearest neighbour, restricted
/// to the translated bounding box of the source mask.
/// Carry a mask from one lattice to another through a transform.
///
/// The bounding box is mapped forward to find the destination box worth
/// filling; every voxel of that box is then mapped *back* and takes the
/// value it lands on, which is what keeps the result free of holes whatever
/// the two spacings are. A pure shift and a hand-typed matrix are the same
/// operation here, and go through the same code.
fn map_mask(
    mask: &[u8],
    from: &crate::volume::Grid,
    to: &crate::volume::Grid,
    t: &Transform3,
) -> Vec<u8> {
    let [nx, ny, nz] = to.dims;
    let mut out = vec![0u8; nx * ny * nz];
    // Bounding box of the source mask, in source voxels.
    let [sx, sy, sz] = from.dims;
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for k in 0..sz {
        for j in 0..sy {
            for i in 0..sx {
                if mask[k * sx * sy + j * sx + i] != 0 {
                    let v = [i, j, k];
                    for a in 0..3 {
                        lo[a] = lo[a].min(v[a]);
                        hi[a] = hi[a].max(v[a]);
                    }
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        return out;
    }
    // The eight translated corners, in destination voxels, give the
    // destination box to fill (padded a voxel for rounding).
    let (mut dlo, mut dhi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &ci in &[lo[0], hi[0]] {
        for &cj in &[lo[1], hi[1]] {
            for &ck in &[lo[2], hi[2]] {
                let p = t.map(from.voxel_to_patient(ci as f64, cj as f64, ck as f64));
                let v = to.patient_to_voxel(p);
                for a in 0..3 {
                    dlo[a] = dlo[a].min(v[a]);
                    dhi[a] = dhi[a].max(v[a]);
                }
            }
        }
    }
    let clamp = |v: f64, n: usize| (v.max(0.0) as usize).min(n.saturating_sub(1));
    let (blo, bhi) = (
        [
            clamp(dlo[0].floor() - 1.0, nx),
            clamp(dlo[1].floor() - 1.0, ny),
            clamp(dlo[2].floor() - 1.0, nz),
        ],
        [
            clamp(dhi[0].ceil() + 1.0, nx),
            clamp(dhi[1].ceil() + 1.0, ny),
            clamp(dhi[2].ceil() + 1.0, nz),
        ],
    );
    for k in blo[2]..=bhi[2] {
        for j in blo[1]..=bhi[1] {
            for i in blo[0]..=bhi[0] {
                let p = t.unmap(to.voxel_to_patient(i as f64, j as f64, k as f64));
                let v = from.patient_to_voxel(p);
                let (si, sj, sk) = (v[0].round(), v[1].round(), v[2].round());
                if si < 0.0 || sj < 0.0 || sk < 0.0 {
                    continue;
                }
                let (si, sj, sk) = (si as usize, sj as usize, sk as usize);
                if si >= sx || sj >= sy || sk >= sz {
                    continue;
                }
                if mask[sk * sx * sy + sj * sx + si] != 0 {
                    out[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::volume::Grid;

    fn grid(origin: Vec3) -> Grid {
        Grid {
            dims: [20, 20, 10],
            spacing: [1.0, 1.0, 2.0],
            origin,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        }
    }

    #[test]
    fn a_translated_mask_lands_at_the_offset_position() {
        let g1 = grid(Vec3::ZERO);
        let g2 = grid(Vec3::new(2.0, 0.0, 0.0)); // destination shifted lattice
        let mut m = vec![0u8; 20 * 20 * 10];
        // A 3×3×1 block around voxel (5, 5, 5).
        for j in 4..7 {
            for i in 4..7 {
                m[5 * 400 + j * 20 + i] = 1;
            }
        }
        let delta = Vec3::new(6.0, -2.0, 0.0);
        let out = map_mask(
            &m,
            &g1,
            &g2,
            &Transform3::from_matrix(crate::registration::Mat4::translation(delta), Vec3::ZERO),
        );
        let c_in = crate::motion::centroid_mm(&m, &g1).unwrap();
        let c_out = crate::motion::centroid_mm(&out, &g2).unwrap();
        let moved = c_out - c_in;
        assert!((moved - delta).length() < 0.75, "moved {moved:?}");
        assert_eq!(
            out.iter().filter(|&&v| v != 0).count(),
            9,
            "the block keeps its size"
        );
    }
}
