//! *Tools ▶ Transfer by relationship*: place a structure into the other
//! dataset by its spatial relationship to a reference structure.
//!
//! A STAR target defined on one patient's imaging cannot be propagated onto
//! another patient (or another posture) by registration alone when the two
//! datasets share no anatomy-to-anatomy correspondence for it. What travels
//! instead is the *relationship*: the target's offset from the centroid of a
//! reference structure both datasets can segment — typically the heart. The
//! target lands in the destination at the same offset from the destination's
//! reference structure, keeping its shape; deformable adaptation, when
//! wanted, is the propagation tool's job afterwards.

use crate::motion;

use super::*;

/// The window's state.
pub(super) struct TransferDialog {
    /// Dataset the target comes from; it lands on the other one.
    pub src_slot: usize,
    /// Candidate index of the target in the source dataset.
    pub target: Option<usize>,
    /// Candidate index of the reference structure in the source dataset.
    pub src_ref: Option<usize>,
    /// Candidate index of the reference structure in the destination.
    pub dst_ref: Option<usize>,
    pub status: Option<String>,
}

impl ViewerApp {
    pub(super) fn open_transfer_dialog(&mut self, src_slot: usize) {
        let mut d = TransferDialog {
            src_slot,
            target: None,
            src_ref: None,
            dst_ref: None,
            status: None,
        };
        // Pre-pick reference structures by the obvious name.
        let guess = |cands: &[(super::combine_win::ItemRef, String)]| {
            cands.iter().position(|(_, l)| {
                let l = l.to_lowercase();
                l.contains("heart") || l.contains("herz")
            })
        };
        d.src_ref = guess(&self.combine_candidates(src_slot));
        d.dst_ref = guess(&self.combine_candidates(1 - src_slot));
        self.transfer_dialog = Some(d);
    }

    /// Carry the target across, synchronously — a translation and one
    /// nearest-neighbour resampling over the target's bounding box.
    fn transfer_now(&mut self) {
        let Some(d) = &self.transfer_dialog else {
            return;
        };
        let (src, dst) = (d.src_slot, 1 - d.src_slot);
        let pick = |slot: usize, sel: Option<usize>| {
            sel.and_then(|i| self.combine_candidates(slot).get(i).cloned())
        };
        let (Some((it, _)), Some((ir, _)), Some((id_, _))) = (
            pick(src, d.target),
            pick(src, d.src_ref),
            pick(dst, d.dst_ref),
        ) else {
            if let Some(d) = &mut self.transfer_dialog {
                d.status = Some("Pick the target and both reference structures first.".into());
            }
            return;
        };
        let (Some((tm, tg, tname, tcolor)), Some((rm, rg, rname, _)), Some((dm, dg, dname, _))) = (
            self.item_mask_grid(src, it),
            self.item_mask_grid(src, ir),
            self.item_mask_grid(dst, id_),
        ) else {
            if let Some(d) = &mut self.transfer_dialog {
                d.status = Some("One of the structures is gone or empty.".into());
            }
            return;
        };
        let (Some(c_target), Some(c_src), Some(c_dst)) = (
            motion::centroid_mm(&tm, &tg),
            motion::centroid_mm(&rm, &rg),
            motion::centroid_mm(&dm, &dg),
        ) else {
            if let Some(d) = &mut self.transfer_dialog {
                d.status = Some("One of the structures has no voxels.".into());
            }
            return;
        };
        let delta = c_dst - c_src;

        // The destination lattice is the displayed volume of the other
        // dataset — that is where a new segmentation is editable.
        let Some(study) = self.slots[dst].study.as_ref() else {
            return;
        };
        let out_grid = study.volume.grid();
        let mask = translate_mask(&tm, &tg, &out_grid, delta);
        if mask.iter().all(|&v| v == 0) {
            if let Some(dlg) = &mut self.transfer_dialog {
                dlg.status = Some(format!(
                    "'{tname}' lands outside dataset {}'s displayed volume — nothing to store.",
                    SLOT_NAMES[dst]
                ));
            }
            return;
        }
        let placed_cm3 = motion::volume_cm3(&mask, &out_grid);
        let name = format!("{tname} @ {dname}");
        let dims = out_grid.dims;
        self.add_colored_segmentation(dst, name.clone(), tcolor, dims, &mask);
        if let Some(dlg) = &mut self.transfer_dialog {
            dlg.status = Some(format!(
                "'{name}' stored in dataset {} — offset from {rname}: RL {:+.1} · AP {:+.1} · \
                 SI {:+.1} mm, {placed_cm3:.2} cm³.",
                SLOT_NAMES[dst],
                c_target.x - c_src.x,
                c_target.y - c_src.y,
                c_target.z - c_src.z,
            ));
        }
    }

    pub(super) fn transfer_window(&mut self, ctx: &egui::Context) {
        let Some(d) = &self.transfer_dialog else {
            return;
        };
        let src = d.src_slot;
        let dst = 1 - src;
        if self.slots[src].study.is_none() || self.slots[dst].study.is_none() {
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
        let mut run = false;
        let mut close = false;
        let mut swap = false;
        let mut open = true;
        let d = self.transfer_dialog.as_mut().expect("checked above");
        detach::tool_window(
            ctx,
            "transfer",
            "◎ Transfer by relationship",
            &mut open,
            detach::WinOpts::default().resizable(false),
            |ui| {
                ui.label(format!(
                    "Place a structure of dataset {} into dataset {} at the same offset \
                     from a reference structure (e.g. the heart) — the target–reference \
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
                        let sel = item
                            .and_then(|i| list.get(i).cloned())
                            .unwrap_or_else(|| "(pick)".into());
                        egui::ComboBox::from_id_salt(salt.to_string())
                            .width(260.0)
                            .selected_text(sel)
                            .show_ui(ui, |ui| {
                                for (i, l) in list.iter().enumerate() {
                                    ui.selectable_value(item, Some(i), l);
                                }
                            });
                    });
                };
                combo(
                    ui,
                    &format!("Target ({}):", SLOT_NAMES[src]),
                    &mut d.target,
                    &src_cands,
                    "tr_target",
                );
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
                if ui
                    .button(format!("Swap direction (to dataset {})", SLOT_NAMES[src]))
                    .clicked()
                {
                    swap = true;
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
        if swap {
            if let Some(d) = &mut self.transfer_dialog {
                d.src_slot = 1 - d.src_slot;
                d.target = None;
                d.src_ref = None;
                d.dst_ref = None;
                d.status = None;
            }
            if let Some(slot) = self.transfer_dialog.as_ref().map(|d| d.src_slot) {
                let guess = |cands: Vec<(super::combine_win::ItemRef, String)>| {
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
fn translate_mask(
    mask: &[u8],
    from: &crate::volume::Grid,
    to: &crate::volume::Grid,
    delta: crate::geometry::Vec3,
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
                let p = from.voxel_to_patient(ci as f64, cj as f64, ck as f64) + delta;
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
                let p = to.voxel_to_patient(i as f64, j as f64, k as f64) - delta;
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
        let out = translate_mask(&m, &g1, &g2, delta);
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
