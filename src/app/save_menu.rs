//! Everything that writes something out, in one place: the *Save* menu of
//! the menu bar, the *Save* entries of the data tree's right-click menus,
//! and the two ways a planar image goes out with its 2D contours drawn in
//! (a DICOM Secondary Capture, a PNG).
//!
//! **One way per kind of output.** The menu and the tree only decide *what*
//! is saved; how is the window or the dialog that already does it: *Export
//! DICOM* for anything DICOM (opened with only the object asked for ticked,
//! [`crate::export::ExportPlan::select_only`], so its names and UIDs are on
//! screen before anything is written), *Save image* for a picture of the
//! screen (a row, all rows, or one pane), *Save to the PACS*, and the
//! presentation state of a planar structure set. The tree's entries are
//! requests ([`SaveRequest`]) carried out once the tree is drawn, as its
//! renames and removals are.
//!
//! **Contours burned in.** A presentation state is the faithful form of 2D
//! contours, and many viewers never draw one. A planar image can therefore
//! also be saved with its contours drawn into the pixels
//! ([`crate::burn_in`]): what is drawn is what the panes draw - the
//! structures of every planar set on that image that are not hidden in the
//! tree - through the window a pane shows the image with.

use crate::export::{ObjKind as XKind, Pick};

use super::*;

/// What a menu entry asked to have saved.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum SaveRequest {
    /// Open *Export DICOM* with this one ticked.
    Dicom(Pick),
    /// Planar image `idx` with its 2D contours burned in, as a Secondary
    /// Capture.
    PlanarBurned { slot: usize, idx: usize },
    /// The same as a PNG at the image's own resolution.
    PlanarPng { slot: usize, idx: usize },
    /// Planar structure set `idx` as a presentation state.
    PlanarSet { slot: usize, idx: usize },
    /// *Save image* on one pane.
    Pane { slot: usize, kind: PaneKind },
}

const DICOM: pick::Filter = pick::Filter {
    name: "DICOM",
    exts: &["dcm"],
};
const PNG: pick::Filter = pick::Filter {
    name: "PNG",
    exts: &["png"],
};

impl ViewerApp {
    /// The entry every data-tree menu carries for its object: *Export
    /// DICOM* with only that one ticked.
    pub(super) fn save_dicom_entry(ui: &mut egui::Ui, what: &str) -> bool {
        ui.button("💾 Save as DICOM...")
            .on_hover_text(format!(
                "Export DICOM with only this {what} ticked: its names and UIDs shown and \
                 editable, then one click on Export"
            ))
            .clicked()
    }

    /// The *Save* entries of a planar image's menu.
    pub(super) fn planar_save_entries(
        &self,
        ui: &mut egui::Ui,
        slot: usize,
        idx: usize,
        out: &mut Option<SaveRequest>,
    ) {
        let projection = self.planar_kind(slot, idx) == Some(crate::extras::PlanarKind::Projection);
        let lines = !self.planar_lines(slot, idx).is_empty();
        ui.menu_button("💾 Save", |ui| {
            if enabled_tip_button(
                ui,
                projection,
                "💾 As DICOM...",
                if projection {
                    "Export DICOM with this image ticked, and the planar structure sets drawn \
                     on it with it: the image, its contours as a presentation state, and a \
                     copy with the contours burned in"
                } else {
                    "An additional object is not written by Export DICOM; its own file is \
                     the one to keep"
                },
            ) {
                *out = Some(SaveRequest::Dicom(Pick::Object {
                    slot,
                    kind: XKind::Planar,
                    index: idx,
                }));
                ui.close();
            }
            let none = "no 2D contours on this image (hidden ones are not drawn)";
            if enabled_tip_button(
                ui,
                lines,
                "🖊 With contours burned in (DICOM)...",
                if lines {
                    "One file: the image as the pane shows it, with its 2D contours drawn into \
                     the pixels in their colours - a Secondary Capture any viewer shows, \
                     including those that draw no presentation state"
                } else {
                    none
                },
            ) {
                *out = Some(SaveRequest::PlanarBurned { slot, idx });
                ui.close();
            }
            if tip_button(
                ui,
                "🎨 As PNG...",
                "The image at its own resolution, as the pane shows it, with its 2D contours \
                 drawn in",
            ) {
                *out = Some(SaveRequest::PlanarPng { slot, idx });
                ui.close();
            }
        });
    }

    /// The *Save* menu of the menu bar. Returns what was asked for, and
    /// sets the flags of the entries that open a window of their own.
    pub(super) fn save_menu(
        &self,
        ui: &mut egui::Ui,
        open_export: &mut bool,
        open_save_img: &mut bool,
        save_to_pacs: &mut Option<usize>,
    ) -> Option<SaveRequest> {
        let mut req = None;
        let loaded: Vec<usize> = (0..MAX_WORKSPACES)
            .filter(|s| self.slots[*s].study.is_some())
            .collect();
        let anything = !loaded.is_empty();
        if enabled_tip_button(
            ui,
            anything,
            "💾 Export DICOM",
            "Write any patients, studies, series and RT objects of the workspaces as DICOM - \
             with every name and UID shown and editable, structures as RTSTRUCT or SEG, 2D \
             contours as presentation states and burned into copies of their images, and \
             the references between the objects kept intact",
        ) {
            *open_export = true;
            ui.close();
        }
        ui.add_enabled_ui(anything, |ui| {
            ui.menu_button("📤 Save to the PACS", |ui| {
                for slot in &loaded {
                    let label =
                        format!("{} · {}", SLOT_NAMES[*slot], self.workspace_summary(*slot));
                    if ui
                        .button(label)
                        .on_hover_text(
                            "Write this workspace's new structure sets and segmentations into \
                             this station's PACS, beside the study they belong to, in the \
                             PACS's own folders",
                        )
                        .clicked()
                    {
                        *save_to_pacs = Some(*slot);
                        ui.close();
                    }
                }
            });
        });
        ui.separator();
        if tip_button(
            ui,
            "💾 Save image",
            "A picture of the views as they are on screen - one workspace's row, all of \
             them, or one pane - as a PNG or a JPEG, at a chosen resolution",
        ) {
            *open_save_img = true;
            ui.close();
        }
        let panes = self.panes_on_screen();
        ui.add_enabled_ui(!panes.is_empty(), |ui| {
            ui.menu_button("⛶ Image of one pane", |ui| {
                for (slot, kind, _) in &panes {
                    if ui
                        .button(self.pane_caption(*slot, *kind))
                        .on_hover_text("This pane on its own, in the Save image dialog")
                        .clicked()
                    {
                        req = Some(SaveRequest::Pane {
                            slot: *slot,
                            kind: *kind,
                        });
                        ui.close();
                    }
                }
            });
        });
        // The 2D contours, when there are any.
        let sets: Vec<(usize, usize, String)> = loaded
            .iter()
            .flat_map(|&slot| {
                self.slots[slot].study.iter().flat_map(move |st| {
                    st.planar_structs
                        .iter()
                        .enumerate()
                        .map(move |(k, s)| (slot, k, s.label.clone()))
                })
            })
            .collect();
        let drawn: Vec<(usize, usize, String)> = loaded
            .iter()
            .flat_map(|&slot| {
                let n = self.planar_count(slot);
                (0..n)
                    .filter(move |&i| !self.planar_lines(slot, i).is_empty())
                    .map(move |i| (slot, i, self.planar_label(slot, i)))
            })
            .collect();
        if !sets.is_empty() || !drawn.is_empty() {
            ui.separator();
        }
        let many = loaded.len() > 1;
        let name = |slot: usize, label: &str| {
            if many {
                format!("{} · {label}", SLOT_NAMES[slot])
            } else {
                label.to_string()
            }
        };
        if !sets.is_empty() {
            ui.menu_button("◌ 2D contours as presentation state", |ui| {
                for (slot, k, label) in &sets {
                    if ui
                        .button(name(*slot, label))
                        .on_hover_text(
                            "A Grayscale Softcopy Presentation State, with every image made \
                             here as an RT Image beside it",
                        )
                        .clicked()
                    {
                        req = Some(SaveRequest::PlanarSet {
                            slot: *slot,
                            idx: *k,
                        });
                        ui.close();
                    }
                }
            });
        }
        if !drawn.is_empty() {
            ui.menu_button("🖊 Planar image with contours burned in", |ui| {
                for (slot, i, label) in &drawn {
                    if ui
                        .button(name(*slot, label))
                        .on_hover_text(
                            "The image as its pane shows it, with its 2D contours drawn into \
                             the pixels: a DICOM Secondary Capture any viewer shows",
                        )
                        .clicked()
                    {
                        req = Some(SaveRequest::PlanarBurned {
                            slot: *slot,
                            idx: *i,
                        });
                        ui.close();
                    }
                }
            });
        }
        ui.separator();
        ui.weak("Right-click an object in the data tree to save that one alone.");
        req
    }

    /// Carry out a [`SaveRequest`].
    pub(super) fn apply_save_request(&mut self, req: SaveRequest) {
        match req {
            SaveRequest::Dicom(pick) => self.open_export_only(pick),
            SaveRequest::PlanarBurned { slot, idx } => self.save_planar_drawn(slot, idx, false),
            SaveRequest::PlanarPng { slot, idx } => self.save_planar_drawn(slot, idx, true),
            SaveRequest::PlanarSet { slot, idx } => self.save_planar_set(slot, idx),
            SaveRequest::Pane { slot, kind } => {
                self.save_img = Some(snapshot::SaveImgDialog {
                    what: snapshot::ImgWhat::Pane(slot, kind),
                    ..Default::default()
                });
                self.raise_window("save_image");
            }
        }
    }

    /// *Export DICOM* with only `pick` ticked.
    pub(super) fn open_export_only(&mut self, pick: Pick) {
        self.open_export_dialog();
        let studies: [Option<&LoadedStudy>; MAX_WORKSPACES] =
            std::array::from_fn(|s| self.slots[s].study.as_ref());
        let Some(plan) = self.export_plan.as_mut() else {
            return;
        };
        if !plan.select_only(&pick, &studies) {
            self.notice = Some(
                "That object is not one Export DICOM writes; the window is open with \
                 everything ticked"
                    .into(),
            );
        }
    }

    /// The label of planar image `idx`, for a menu line.
    fn planar_label(&self, slot: usize, idx: usize) -> String {
        self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_images.get(idx))
            .map(|p| {
                if p.label.trim().is_empty() {
                    format!("{} image {}", p.modality, idx + 1)
                } else {
                    p.label.clone()
                }
            })
            .unwrap_or_default()
    }

    /// The 2D contours the panes draw on planar image `idx`: the structures
    /// of every planar set with a contour or a point on it, less the ones
    /// hidden in the tree.
    pub(super) fn planar_lines(&self, slot: usize, idx: usize) -> Vec<crate::burn_in::Overlay> {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return Vec::new();
        };
        let Some(img) = study.planar_images.get(idx) else {
            return Vec::new();
        };
        crate::burn_in::overlays(&study.planar_structs, &img.sop_instance_uid, &|set, s| {
            self.planar_visible(slot, set, &s.name)
        })
    }

    /// The window planar image `idx` is shown with: a pane's, where one
    /// shows it, else its own.
    pub(super) fn planar_shown_window(&self, slot: usize, idx: usize) -> Option<(f32, f32)> {
        for k in 0..super::planar::PANE_VIEWS {
            for (li, layer) in self.slots[slot].planar[k].layers.iter().enumerate() {
                if layer.image == idx {
                    if let Some((_, wl)) = self.planar_wl((slot, k, li)) {
                        return Some(wl);
                    }
                }
            }
        }
        let img = self.slots[slot].study.as_ref()?.planar_images.get(idx)?;
        Some(super::planar::default_window(img))
    }

    /// Write planar image `idx` with its contours burned in: a Secondary
    /// Capture, or with `png` a PNG.
    fn save_planar_drawn(&mut self, slot: usize, idx: usize, png: bool) {
        let Some(wl) = self.planar_shown_window(slot, idx) else {
            return;
        };
        let lines = self.planar_lines(slot, idx);
        let Some(img) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_images.get(idx))
        else {
            return;
        };
        if img.rows * img.cols == 0 {
            self.error = Some("The image has no pixels to save".into());
            return;
        }
        let rgb = crate::burn_in::render(img, wl, &lines);
        // A file name says nothing of the image: its label may be anything,
        // a patient's name included.
        let (title, name, filter) = if png {
            ("Save the image as a PNG", "planar_image.png", PNG)
        } else {
            (
                "Save the image with its contours burned in",
                "planar_contours.dcm",
                DICOM,
            )
        };
        self.ask_save(title, name, None, Some(filter), move |app, dest| {
            let Some(study) = app.slots[slot].study.as_ref() else {
                return;
            };
            let Some(img) = study.planar_images.get(idx) else {
                return;
            };
            let res = if png {
                let picture = egui::ColorImage::from_rgb([img.cols, img.rows], &rgb);
                snapshot::write_png(&dest, &picture, 96).map(|_| ())
            } else {
                let meta = crate::dicom_export::ObjectMeta::of_study(
                    study,
                    "CONTOURS",
                    &format!("{} with contours", img.label),
                );
                crate::burn_in::write_secondary_capture(&dest, img, &rgb, &meta, "", 900)
                    .map(|_| ())
            };
            match res {
                Ok(()) => app.notice = Some(format!("Written to {}", dest.display())),
                Err(e) => app.error = Some(format!("Could not write the image: {e:#}")),
            }
        });
    }
}
