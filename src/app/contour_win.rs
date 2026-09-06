//! *Tools ▶ Contour tools*: everything that acts on a whole structure
//! rather than on the stroke under the pointer.
//!
//! The drawing tools in the toolbar put contours on a slice. This window
//! does the rest of what a planner spends the day on: interpolating the
//! slices that were skipped, tidying what a rushed hand left behind, moving
//! a structure that is right in shape and wrong in place, and thinning a
//! geometry back out so it can be re-interpolated.
//!
//! Everything here edits *the structure the contour tools edit* - the one
//! marked with ✏ in the RT structures list - and every button is one undo
//! step (Ctrl+Z with a contour tool in hand).

use super::seg_engines::ToolInfo;
use super::*;

/// The seventh tool. A memo, because this is the window that edits what is
/// already written; the glyph is one egui's bundled fonts carry (see the
/// `glyphs` guard).
pub(super) const CONTOURS: ToolInfo = ToolInfo {
    glyph: "📝",
    name: "Contour tools",
    verb: "Edit contours in",
};

/// The RT ROI Interpreted Types offered when retyping a structure. The list
/// is the one a planning system branches on; anything else can still arrive
/// from a file and is left alone.
const ROI_TYPES: &[&str] = &[
    "ORGAN",
    "PTV",
    "CTV",
    "GTV",
    "EXTERNAL",
    "AVOIDANCE",
    "BOLUS",
    "SUPPORT",
    "FIXATION",
    "MARKER",
    "CONTROL",
];

pub(super) struct ContourDialog {
    pub slot: usize,
    /// Show the interpolated slices as a dashed preview.
    pub show_interp: bool,
    /// Tidy: contours smaller than this (cm²) go.
    pub min_area_cm2: f32,
    /// Tidy: cap on the points of one contour.
    pub max_points: usize,
    /// Tidy: smoothing passes.
    pub smooth: usize,
    /// Thin: keep every n-th slice.
    pub keep_nth: usize,
    /// Move / scale / rotate, in the plane of the drawing axis.
    pub shift: [f32; 2],
    pub scale_pct: f32,
    pub rotate_deg: f32,
}

impl ContourDialog {
    fn new(slot: usize) -> ContourDialog {
        ContourDialog {
            slot,
            show_interp: false,
            min_area_cm2: 0.05,
            max_points: 200,
            smooth: 1,
            keep_nth: 2,
            shift: [0.0, 0.0],
            scale_pct: 100.0,
            rotate_deg: 0.0,
        }
    }
}

impl ViewerApp {
    pub(super) fn open_contour_dialog(&mut self, slot: usize) {
        self.contour_dialog = Some(ContourDialog::new(slot));
    }

    pub(super) fn contour_window(&mut self, ctx: &egui::Context) {
        if self.contour_dialog.is_none() {
            return;
        }
        let slot = self.contour_dialog.as_ref().map(|d| d.slot).unwrap_or(0);
        // The preview is recomputed here, once a frame, and only while it is
        // wanted: it is the one thing in this window that costs anything.
        if self.contour_dialog.as_ref().is_some_and(|d| d.show_interp) {
            self.refresh_interp(slot);
        } else if self.interp.is_some() {
            self.interp = None;
        }

        self.refresh_derived(slot);
        let target = self.edit_target(slot);
        let derived = target.and_then(|(set, roi)| self.derived_of(slot, set, roi));
        let derived_line = derived
            .as_ref()
            .zip(self.edit_roi_name(slot))
            .map(|(d, (n, _))| d.line(n));
        let derived_status = target
            .and_then(|(_, roi)| self.derived_status(slot, roi))
            .map(|s| (s.glyph(), s.label(), s.color()));
        let derived_busy = self.derived_job.is_some();
        let summary = self.edit_summary(slot);
        let axis = self.edit_axis(slot);
        let level = self.edit_level(slot);
        let n_interp = self.interp_count(slot);
        let has_clip = self.contour_clip.is_some();
        let comparison = self.comparison;
        let slices = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.volume.dims[axis])
            .unwrap_or(1);
        let mode = self.draw_mode;

        let mut open = true;
        let mut act: Option<Act> = None;
        let mut new_slot: Option<usize> = None;
        let d = self.contour_dialog.as_mut().expect("checked above");
        detach::tool_window(
            ctx,
            "contour_tools",
            format!("{} {}", CONTOURS.glyph, CONTOURS.name),
            &mut open,
            detach::WinOpts::width(400.0),
            |ui| {
                if comparison {
                    ui.horizontal(|ui| {
                        ui.label("Dataset:");
                        for (s, name) in SLOT_NAMES.iter().enumerate() {
                            if ui.selectable_label(d.slot == s, *name).clicked() {
                                new_slot = Some(s);
                            }
                        }
                    });
                }
                match &summary {
                    Some((name, roi_type, vol, occupied, points)) => {
                        ui.label(egui::RichText::new(format!("✏ {name}")).strong().size(15.0));
                        ui.label(
                            egui::RichText::new(format!(
                                "{roi_type} · {vol:.1} cm³ · {occupied} slice(s) · \
                                 {points} points · drawn on {} slices",
                                crate::contours::axis_name(axis)
                            ))
                            .weak(),
                        );
                    }
                    None => {
                        ui.label(
                            "No structure is being edited. Pick one with the ✏ button in \
                             the RT structures list, Ctrl-click a contour in a view, or \
                             just start drawing - the first stroke creates one.",
                        );
                        return;
                    }
                }
                // -- derived -------------------------------------------
                if let (Some(line), Some((glyph, label, c))) = (&derived_line, &derived_status) {
                    ui.separator();
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new("Derived").strong());
                        ui.label(
                            egui::RichText::new(format!("{glyph} {label}"))
                                .color(egui::Color32::from_rgb(c[0], c[1], c[2])),
                        );
                    });
                    ui.label(egui::RichText::new(line.clone()).italics());
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .add_enabled(!derived_busy, egui::Button::new("Update"))
                            .on_hover_text("Run the recipe again on the current operands")
                            .clicked()
                        {
                            act = Some(Act::DerivedUpdate);
                        }
                        if ui
                            .button("Edit recipe")
                            .on_hover_text("Open it in the Combine window")
                            .clicked()
                        {
                            act = Some(Act::DerivedEdit);
                        }
                        if ui
                            .button("Underive")
                            .on_hover_text(
                                "Forget the recipe and keep the geometry: an ordinary \
                                 structure from here on",
                            )
                            .clicked()
                        {
                            act = Some(Act::Underive);
                        }
                    });
                }
                ui.separator();

                // -- interpolation -------------------------------------
                ui.label(egui::RichText::new("Interpolation").strong());
                ui.label(
                    egui::RichText::new(
                        "Contours for the slices between the drawn ones, blended through \
                         their distance fields. Shown dashed until accepted - nothing is \
                         stored before that.",
                    )
                    .weak(),
                );
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut d.show_interp, "Show");
                    ui.label(format!("{n_interp} slice(s)"));
                    if ui
                        .add_enabled(n_interp > 0, egui::Button::new("Accept this slice"))
                        .on_hover_text(format!(
                            "Slice {level} of {slices} - the one the {} view shows",
                            crate::contours::axis_name(axis)
                        ))
                        .clicked()
                    {
                        act = Some(Act::AcceptInterp(false));
                    }
                    if ui
                        .add_enabled(n_interp > 0, egui::Button::new("Accept all"))
                        .clicked()
                    {
                        act = Some(Act::AcceptInterp(true));
                    }
                });
                ui.add_space(6.0);

                // -- this slice ----------------------------------------
                ui.label(egui::RichText::new(format!("Slice {level} of {slices}")).strong());
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("Copy")
                        .on_hover_text("Copy this slice's contours")
                        .clicked()
                    {
                        act = Some(Act::Copy);
                    }
                    if ui
                        .add_enabled(has_clip, egui::Button::new("Paste"))
                        .on_hover_text(format!(
                            "Paste them here ({} mode)",
                            mode.label().to_lowercase()
                        ))
                        .clicked()
                    {
                        act = Some(Act::Paste);
                    }
                    if ui
                        .button("Delete one")
                        .on_hover_text(
                            "Delete the single contour the crosshair is inside, on this \
                             slice - the rest of the slice is left alone",
                        )
                        .clicked()
                    {
                        act = Some(Act::DeleteOne);
                    }
                    if ui
                        .button("Clear")
                        .on_hover_text("Delete every contour on this slice")
                        .clicked()
                    {
                        act = Some(Act::Clear);
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Thin: keep every");
                    ui.add(egui::DragValue::new(&mut d.keep_nth).range(2..=20));
                    ui.label("th slice");
                    if ui
                        .button("Apply")
                        .on_hover_text(
                            "Drop the slices in between, over the whole structure. What \
                             interpolation is for: thin out, correct two slices, \
                             interpolate again",
                        )
                        .clicked()
                    {
                        act = Some(Act::Thin);
                    }
                });
                ui.add_space(6.0);

                // -- tidying -------------------------------------------
                ui.label(egui::RichText::new("Tidy").strong());
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("Resolve overlaps")
                        .on_hover_text(
                            "Contours of this structure that cross each other become one \
                             clean set of nested rings; the filled area does not change",
                        )
                        .clicked()
                    {
                        act = Some(Act::ResolveOverlaps);
                    }
                    if ui
                        .button("Remove holes")
                        .on_hover_text("Fill every interior cavity, on every slice")
                        .clicked()
                    {
                        act = Some(Act::RemoveHoles);
                    }
                    if ui
                        .button("Keep largest")
                        .on_hover_text("Keep the largest contour of each slice, with its holes")
                        .clicked()
                    {
                        act = Some(Act::KeepLargest);
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Drop contours under");
                    ui.add(
                        egui::DragValue::new(&mut d.min_area_cm2)
                            .speed(0.01)
                            .range(0.0..=100.0)
                            .suffix(" cm²"),
                    );
                    if ui.button("Apply").clicked() {
                        act = Some(Act::DropSmall);
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("At most");
                    ui.add(egui::DragValue::new(&mut d.max_points).range(8..=2000));
                    ui.label("points per contour");
                    if ui
                        .button("Apply")
                        .on_hover_text(
                            "Simplified with the largest tolerance that still fits the \
                             cap, so the shape is kept and the vertices are the \
                             original ones",
                        )
                        .clicked()
                    {
                        act = Some(Act::LimitPoints);
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.add(egui::DragValue::new(&mut d.smooth).range(1..=10));
                    ui.label("smoothing pass(es)");
                    if ui
                        .button("Apply")
                        .on_hover_text("Area-preserving: the outline relaxes, the volume stays")
                        .clicked()
                    {
                        act = Some(Act::Smooth);
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Piece under the crosshair:");
                    if ui
                        .button("Keep it alone")
                        .on_hover_text(
                            "Keep the connected piece the crosshair is in and drop every \
                             other one",
                        )
                        .clicked()
                    {
                        act = Some(Act::Component(true));
                    }
                    if ui
                        .button("Delete it")
                        .on_hover_text("Delete that piece and keep the rest")
                        .clicked()
                    {
                        act = Some(Act::Component(false));
                    }
                });
                ui.add_space(6.0);

                // -- transforms ----------------------------------------
                ui.label(egui::RichText::new("Move the whole structure").strong());
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut d.shift[0])
                            .speed(0.5)
                            .range(-500.0..=500.0)
                            .prefix("x ")
                            .suffix(" mm"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut d.shift[1])
                            .speed(0.5)
                            .range(-500.0..=500.0)
                            .prefix("y ")
                            .suffix(" mm"),
                    );
                    if ui
                        .button("Move")
                        .on_hover_text("In the plane it is drawn on, in millimetres")
                        .clicked()
                    {
                        act = Some(Act::Translate);
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut d.scale_pct)
                            .speed(0.5)
                            .range(10.0..=400.0)
                            .suffix(" %"),
                    );
                    if ui
                        .button("Scale")
                        .on_hover_text("About the structure's own centroid")
                        .clicked()
                    {
                        act = Some(Act::Scale);
                    }
                    ui.add(
                        egui::DragValue::new(&mut d.rotate_deg)
                            .speed(1.0)
                            .range(-180.0..=180.0)
                            .suffix("°"),
                    );
                    if ui.button("Rotate").clicked() {
                        act = Some(Act::Rotate);
                    }
                    if ui
                        .button("To crosshair")
                        .on_hover_text(
                            "Put the structure's centroid under the crosshair - the \
                             \"move to slice intersection\" of a planning system",
                        )
                        .clicked()
                    {
                        act = Some(Act::ToCrosshair);
                    }
                });
                ui.add_space(6.0);

                // -- identity ------------------------------------------
                if let Some((_, roi_type, _, _, _)) = &summary {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Type:");
                        egui::ComboBox::from_id_salt("roi_type")
                            .selected_text(if roi_type.is_empty() {
                                "(none)"
                            } else {
                                roi_type.as_str()
                            })
                            .show_ui(ui, |ui| {
                                for t in ROI_TYPES {
                                    if ui.selectable_label(roi_type == t, *t).clicked() {
                                        act = Some(Act::Retype(t));
                                    }
                                }
                            });
                        ui.label(egui::RichText::new("what a planning system branches on").weak());
                    });
                }
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(
                        "Every button here is one undo step: Ctrl+Z with a contour tool in \
                         hand.",
                    )
                    .weak(),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "Slices are counted along the {} axis - the plane the structure \
                         is drawn on.",
                        crate::contours::axis_name(axis)
                    ))
                    .weak(),
                );
            },
        );

        if let Some(s) = new_slot {
            if let Some(d) = self.contour_dialog.as_mut() {
                d.slot = s;
            }
            self.interp = None;
        }
        if let Some(a) = act {
            self.apply_contour_act(slot, a);
        }
        if !open {
            self.contour_dialog = None;
            self.interp = None;
        }
    }

    fn apply_contour_act(&mut self, slot: usize, act: Act) {
        let d = self.contour_dialog.as_ref();
        let (min_area, max_points, smooth, keep, shift, scale, rot) = match d {
            Some(d) => (
                d.min_area_cm2 as f64,
                d.max_points,
                d.smooth,
                d.keep_nth,
                d.shift,
                d.scale_pct as f64 / 100.0,
                (d.rotate_deg as f64).to_radians(),
            ),
            None => return,
        };
        let axis = self.edit_axis(slot);
        let spacing = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.volume.spacing)
            .unwrap_or([1.0; 3]);
        let [ua, va] = crate::contours::plane_axes(axis);
        match act {
            Act::AcceptInterp(all) => {
                self.accept_interp(slot, all);
            }
            Act::Copy => {
                if !self.copy_slice_contours(slot) {
                    self.notice = Some("This slice has no contours to copy.".into());
                }
            }
            Act::Paste => {
                self.paste_slice_contours(slot);
            }
            Act::Clear => {
                self.clear_slice_contours(slot);
            }
            Act::DeleteOne => {
                self.delete_contour_at_cursor(slot);
            }
            Act::Thin => {
                let range = self
                    .edit
                    .as_ref()
                    .and_then(|e| e.stack.level_range())
                    .unwrap_or((0, 0));
                self.thin_slices(slot, keep, range.0, range.1);
            }
            Act::ResolveOverlaps => {
                self.with_edit_stack(slot, |st, _| st.resolve_overlaps());
            }
            Act::RemoveHoles => {
                self.with_edit_stack(slot, |st, _| st.remove_holes());
            }
            Act::KeepLargest => {
                self.with_edit_stack(slot, |st, _| {
                    for s in &mut st.slices {
                        s.region.keep_largest();
                    }
                });
            }
            Act::DropSmall => {
                // The threshold is an area on the screen, the contours are in
                // voxels: 1 cm² is 100 mm² divided by the in-plane voxel area.
                let cell = spacing[ua] * spacing[va];
                let min_vox = min_area * 100.0 / cell.max(1e-9);
                self.with_edit_stack(slot, |st, _| st.drop_smaller_than(min_vox));
            }
            Act::LimitPoints => {
                self.with_edit_stack(slot, |st, _| st.limit_points(max_points));
            }
            Act::Smooth => {
                self.with_edit_stack(slot, |st, _| st.smooth(smooth));
            }
            Act::Translate => {
                let d = [
                    shift[0] as f64 / spacing[ua].max(1e-9),
                    shift[1] as f64 / spacing[va].max(1e-9),
                ];
                self.with_edit_stack(slot, |st, _| st.translate(d));
            }
            Act::Scale => {
                self.with_edit_stack(slot, |st, _| st.scale(scale));
            }
            Act::Rotate => {
                self.with_edit_stack(slot, |st, _| st.rotate(rot));
            }
            Act::Component(keep) => {
                self.component_at_cursor(slot, keep);
            }
            Act::ToCrosshair => {
                self.move_to_crosshair(slot);
            }
            Act::DerivedUpdate => {
                if let Some((set, roi)) = self.edit_target(slot) {
                    self.start_derived_update(slot, set, roi);
                }
            }
            Act::DerivedEdit => {
                if let Some((set, roi)) = self.edit_target(slot) {
                    self.edit_derived(slot, set, roi);
                }
            }
            Act::Underive => {
                if let Some((set, roi)) = self.edit_target(slot) {
                    self.underive(slot, set, roi);
                }
            }
            Act::Retype(t) => self.set_edit_roi_type(slot, t),
        }
    }
}

/// What the window asked for, applied after its borrow is released.
enum Act {
    AcceptInterp(bool),
    Copy,
    Paste,
    Clear,
    DeleteOne,
    Thin,
    ResolveOverlaps,
    RemoveHoles,
    KeepLargest,
    DropSmall,
    LimitPoints,
    Smooth,
    Translate,
    Scale,
    Rotate,
    Component(bool),
    ToCrosshair,
    DerivedUpdate,
    DerivedEdit,
    Underive,
    Retype(&'static str),
}
