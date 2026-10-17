//! The context menus of the data tree's sets and items, and the menu that
//! picks where a copy or a move goes.

use super::*;
use crate::workflow::group::Landing;

impl ViewerApp {
    // -- Structure sets and segmentation series ----------------------------

    /// Right-click menu of a series node: what image series it is drawn on,
    /// where it goes, and whether it stays.
    pub(super) fn set_context_menu(
        &self,
        ui: &mut egui::Ui,
        here: SetRef,
        out: &mut Option<SetAction>,
    ) {
        let open = &self.open_slots();
        let holds: [bool; MAX_WORKSPACES] = std::array::from_fn(|s| self.slots[s].study.is_some());
        if ui.button("✏ Rename").clicked() {
            *out = Some(SetAction::Rename(here));
            ui.close();
        }
        ui.separator();
        ui.menu_button("🔗 Connect to image series", |ui| {
            let Some(study) = self.slots[here.slot].study.as_ref() else {
                return;
            };
            let current = match here.kind {
                SetKind::Structures => study
                    .structure_sets
                    .get(here.idx)
                    .map(|s| s.referenced_series_uid.clone()),
                SetKind::Segmentations => study
                    .seg_series
                    .get(here.idx)
                    .map(|s| s.referenced_series_uid.clone()),
            }
            .unwrap_or_default();
            for se in &study.series {
                let label = format!(
                    "{} {} {} ({} sl.)",
                    if se.uid == current { "●" } else { "  " },
                    se.modality,
                    if se.description.is_empty() {
                        "series"
                    } else {
                        &se.description
                    },
                    se.files.len()
                );
                if ui.button(label).clicked() {
                    *out = Some(SetAction::Connect(here, se.uid.clone()));
                    ui.close();
                }
            }
            if here.kind == SetKind::Segmentations {
                ui.separator();
                let free = current.is_empty();
                if ui
                    .button(format!(
                        "{} (no image series: any image of this frame)",
                        if free { "●" } else { "  " }
                    ))
                    .on_hover_text(
                        "Tie the segmentation to no image series. It shows on every image \
                         of the same frame of reference - every phase of a 4D study, say - \
                         resampled onto whichever is displayed.",
                    )
                    .clicked()
                {
                    *out = Some(SetAction::Connect(here, String::new()));
                    ui.close();
                }
            }
        });
        ui.separator();
        // The destination has to hold a study already: a structure set is
        // drawn on an image series, and an empty workspace has none.
        let loaded: Vec<usize> = open
            .iter()
            .copied()
            .filter(|s| *s != here.slot && holds[*s])
            .collect();
        Self::transfer_menu(ui, "series", &loaded, open, |to, op| {
            *out = Some(SetAction::Transfer {
                from: here,
                to,
                copy: op == TreeOp::Copy,
            });
        });
        if here.kind == SetKind::Segmentations {
            ui.separator();
            if tip_button(
                ui,
                "💾 Export as DICOM SEG",
                "Write this series as one DICOM Segmentation file",
            ) {
                *out = Some(SetAction::ExportSeg {
                    set: here,
                    items: Vec::new(),
                });
                ui.close();
            }
        }
        ui.separator();
        if ui.button("🗑 Remove").clicked() {
            *out = Some(SetAction::Remove(here));
            ui.close();
        }
    }

    /// Every structure set and segmentation series of both workspaces, as the
    /// destinations of a *Copy to ▶* / *Move to ▶* submenu - plus the two
    /// "make me a new one" entries, so a transfer never needs preparing.
    pub(super) fn destination_menu(&self, ui: &mut egui::Ui, from: SetRef) -> Option<Destination> {
        let mut picked = None;
        for slot in self.open_slots() {
            let slot_name = SLOT_NAMES[slot];
            let Some(study) = self.slots[slot].study.as_ref() else {
                continue;
            };
            ui.label(egui::RichText::new(format!("Workspace {slot_name}")).strong());
            for (i, ss) in study.structure_sets.iter().enumerate() {
                let here = SetRef {
                    slot,
                    kind: SetKind::Structures,
                    idx: i,
                };
                if here == from {
                    continue;
                }
                let name = if ss.label.is_empty() {
                    &ss.file_name
                } else {
                    &ss.label
                };
                if ui
                    .button(format!("▣ {name} ({} ROIs)", ss.rois.len()))
                    .clicked()
                {
                    picked = Some(Destination::Set(here));
                    ui.close();
                }
            }
            for (i, sr) in study.seg_series.iter().enumerate() {
                let here = SetRef {
                    slot,
                    kind: SetKind::Segmentations,
                    idx: i,
                };
                if here == from {
                    continue;
                }
                if ui
                    .button(format!("✏ {} ({} segments)", sr.label, sr.segs.len()))
                    .clicked()
                {
                    picked = Some(Destination::Set(here));
                    ui.close();
                }
            }
            if ui.button("▣ ➕ a new RT structure set").clicked() {
                picked = Some(Destination::Set(SetRef {
                    slot,
                    kind: SetKind::Structures,
                    idx: SetRef::NEW,
                }));
                ui.close();
            }
            if ui.button("✏ ➕ a new segmentation series").clicked() {
                picked = Some(Destination::Set(SetRef {
                    slot,
                    kind: SetKind::Segmentations,
                    idx: SetRef::NEW,
                }));
                ui.close();
            }
            // Every phase of a 4D group at once: the structure as it is,
            // filed on each phase (no registration - the phases are loaded
            // for their lattices and the structure resampled onto each).
            for (gi, g) in study.fourd_groups.iter().enumerate() {
                let n = g.phase_members().len();
                if g.dissolved || n == 0 {
                    continue;
                }
                for (landing, glyph, what) in [
                    (
                        Landing::Segmentation,
                        "✏",
                        "a segmentation series per phase",
                    ),
                    (Landing::StructureSet, "▣", "each phase's RT structure set"),
                ] {
                    if tip_button(
                        ui,
                        format!("{glyph} ⏱ each phase of {} ({n}): {what}", g.name),
                        "Copies the structure onto every phase of the group as it is, \
                         in patient coordinates, with no registration. Use the \
                         propagation module when the structure should follow the motion.",
                    ) {
                        picked = Some(Destination::Phases {
                            slot,
                            group: gi,
                            landing,
                        });
                        ui.close();
                    }
                }
            }
            ui.separator();
        }
        picked
    }

    /// Right-click menu of one structure / segment. `selection` is what is
    /// ticked in the list: right-clicking a ticked row acts on all of them,
    /// right-clicking an unticked one acts on that row alone.
    pub(super) fn item_context_menu(
        &self,
        ui: &mut egui::Ui,
        from: SetRef,
        clicked: usize,
        label: &str,
        selection: &[usize],
        out: &mut Option<ItemAction>,
    ) {
        let items: Vec<usize> = if selection.len() > 1 && selection.contains(&clicked) {
            selection.to_vec()
        } else {
            vec![clicked]
        };
        let what = if items.len() > 1 {
            format!(
                "the {} ticked {}",
                items.len(),
                from.kind.item_name(items.len())
            )
        } else {
            format!("'{label}'")
        };
        if ui.button("✏ Rename").clicked() {
            *out = Some(ItemAction::Rename { from, idx: clicked });
            ui.close();
        }
        ui.separator();
        ui.menu_button(format!("Copy {what} to"), |ui| {
            if let Some(to) = self.destination_menu(ui, from) {
                *out = Some(to.transfer(from, items.clone(), true));
            }
        });
        ui.menu_button(format!("Move {what} to"), |ui| {
            if let Some(to) = self.destination_menu(ui, from) {
                *out = Some(to.transfer(from, items.clone(), false));
            }
        });
        ui.separator();
        if tip_button(
            ui,
            format!("∪ Combine {what}"),
            "Open the structure-algebra window with these as its operands - union, \
             intersection, subtraction, margins",
        ) {
            *out = Some(ItemAction::Combine {
                from,
                items: items.clone(),
            });
            ui.close();
        }
        if tip_button(
            ui,
            format!("📊 Plot {what} on a DVH"),
            "Open the dose-volume histogram window with these structures against \
             the loaded dose",
        ) {
            *out = Some(ItemAction::Dvh {
                from,
                items: items.clone(),
            });
            ui.close();
        }
        if tip_button(
            ui,
            format!("⇄ Map {what} to the other workspace"),
            "Open the propagation module with these structures picked, to carry them \
             through the active registration - rigid or deformable, landing as \
             contours or as masks",
        ) {
            *out = Some(ItemAction::Map {
                from,
                items: items.clone(),
            });
            ui.close();
        }
        if from.kind == SetKind::Segmentations {
            ui.separator();
            if tip_button(
                ui,
                format!("💾 Export {what} as DICOM SEG"),
                "Writes just these segments, as a SEG series of their own",
            ) {
                *out = Some(ItemAction::ExportSeg {
                    from,
                    items: items.clone(),
                });
                ui.close();
            }
        }
        ui.separator();
        if ui.button("🗑 Remove").clicked() {
            *out = Some(ItemAction::Remove {
                from,
                items: items.clone(),
            });
            ui.close();
        }
    }
}
