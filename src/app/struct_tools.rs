//! *Modules ▶ Structure editor*: everything that makes or changes a whole
//! structure, as one section of the modules panel - and the drawing strip
//! the toolbar's *✏ Draw structure* button unfolds.
//!
//! The editor has three sections:
//!
//! * **Insert structure** - an empty structure, a point of interest, or a
//!   generated one: a grey-level window, a shape, a dose level or the field
//!   of view, landing as an ordinary editable RT structure.
//! * **Edit structure** - everything that acts on the structure selected in
//!   the list rather than on the stroke under the pointer: interpolation,
//!   the slice clipboard, tidying, moving, the type and the derived recipe.
//!   Every button is one undo step (Ctrl+Z with a contour tool in hand).
//! * **Combine structures** - the structure algebra (`combine.rs`).
//!
//! The nine tools that take over the left mouse button are not in the
//! panel: they are the [`ViewerApp::draw_strip`] the toolbar shows while
//! *Draw structure* is on, so a hand that is drawing never leaves the top
//! of the window. The tools that keep a window of their own are the ones
//! that need the room: the engines, Body contour, the details table.

use crate::contours::{axis_name, plane_axes, Stack};
use crate::generate::{self, Shape};
use crate::geometry::Vec3;

use super::combine::ItemRef;
use super::contour_edit::EdgeBand;
use super::*;

/// The RT ROI Interpreted Types offered when typing a structure. The list
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

/// The three sections of the editor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Section {
    Insert,
    Edit,
    Combine,
}

/// Which generator the *Insert structure* section is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Source {
    GreyLevel,
    Shape,
    Dose,
    /// The reconstructed field of view of the displayed images.
    Fov,
}

impl Source {
    const ALL: [Source; 4] = [Source::GreyLevel, Source::Shape, Source::Dose, Source::Fov];
    fn label(self) -> &'static str {
        match self {
            Source::GreyLevel => "Grey level",
            Source::Shape => "Shape",
            Source::Dose => "Dose",
            Source::Fov => "Field of view",
        }
    }
}

/// The generator's settings; they outlive a run, so a second threshold
/// starts from the first.
pub(super) struct NewRoi {
    pub source: Source,
    pub name: String,
    pub roi_type: String,
    // Grey level.
    pub lo: f32,
    pub hi: f32,
    /// PET: read and write the two bounds as SUV rather than as stored
    /// values. Only offered when the displayed series carries what an SUV
    /// takes ([`crate::loader::suv_bw_factor`]).
    pub suv: bool,
    /// Restrict the threshold to a structure that is already drawn.
    pub limit: Option<ItemRef>,
    pub keep_largest: bool,
    pub fill_holes: bool,
    // Shape.
    pub shape: Shape,
    /// Half-sizes along the patient x, y, z, in millimetres.
    pub half: [f32; 3],
    /// Centre of the shape; seeded from the crosshair when the section is
    /// revealed, and again on the ⌖ button.
    pub centre: [f32; 3],
    // Dose.
    pub dose_pct: f32,
    pub dose_absolute: bool,
    pub dose_gy: f32,
    pub status: Option<String>,
}

impl Default for NewRoi {
    fn default() -> Self {
        NewRoi {
            source: Source::GreyLevel,
            name: "New ROI".to_string(),
            roi_type: "ORGAN".to_string(),
            lo: 200.0,
            hi: 4000.0,
            suv: false,
            limit: None,
            keep_largest: false,
            fill_holes: false,
            shape: Shape::Sphere,
            half: [20.0, 20.0, 20.0],
            centre: [0.0; 3],
            dose_pct: 95.0,
            dose_absolute: false,
            dose_gy: 50.0,
            status: None,
        }
    }
}

/// The editor's state: which dataset it works on, and the numbers its
/// buttons apply.
pub(super) struct StructTools {
    /// The dataset the editor acts on.
    pub slot: usize,
    /// A section something asked to open (a context menu, a derived
    /// structure's *Edit recipe*); applied on the next frame the panel is
    /// drawn, then forgotten.
    pub reveal: Option<Section>,
    /// Drawn this frame - when it is not, the interpolation preview is
    /// dropped, because nothing shows what it belongs to.
    pub visible: bool,
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
    pub new: NewRoi,
}

impl Default for StructTools {
    fn default() -> Self {
        StructTools {
            slot: 0,
            reveal: None,
            visible: false,
            show_interp: false,
            min_area_cm2: 0.05,
            max_points: 200,
            smooth: 1,
            keep_nth: 2,
            shift: [0.0, 0.0],
            scale_pct: 100.0,
            rotate_deg: 0.0,
            new: NewRoi::default(),
        }
    }
}

/// What the *Edit* section asked for, applied after its borrow is released.
#[derive(Clone, Copy)]
enum Act {
    AcceptInterp(bool),
    /// Turn the edge snapping of the interpolation on or off.
    SnapInterp(bool),
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

/// A row of small buttons that each ask for one [`Act`]: label, tooltip,
/// enabled, act. The first one clicked lands in `out`.
fn act_row(ui: &mut egui::Ui, out: &mut Option<Act>, buttons: &[(&str, &str, bool, Act)]) {
    for (label, tip, enabled, act) in buttons {
        let r = ui.add_enabled(*enabled, egui::Button::new(*label).small());
        let r = if tip.is_empty() {
            r
        } else {
            r.on_hover_text(*tip)
        };
        if r.clicked() {
            *out = Some(*act);
        }
    }
}

impl ViewerApp {
    /// Switch the editor on, expand the panel and open one section - what
    /// the context menus and a derived structure's *Edit recipe* do.
    pub(super) fn reveal_editor(&mut self, slot: usize, section: Section) {
        self.module_structures = true;
        self.right_open = true;
        self.tools.slot = slot;
        self.tools.reveal = Some(section);
        self.persist_settings();
    }

    /// Open the editor on the structure `roi` of `slot`'s active set.
    pub(super) fn edit_roi_in_editor(&mut self, slot: usize, roi: usize) {
        self.slots[slot].active_roi = roi;
        self.edit = None;
        self.reveal_editor(slot, Section::Edit);
    }

    /// Patient coordinates of the slot's crosshair.
    pub(super) fn crosshair_patient(&self, slot: usize) -> Option<Vec3> {
        let s = &self.slots[slot];
        let v = s.study.as_ref()?;
        let c = s.cursor;
        Some(v.volume.voxel_to_patient(c[0], c[1], c[2]))
    }

    fn seed_shape_centre(&mut self, slot: usize) {
        let c = self.crosshair_patient(slot).unwrap_or(Vec3::ZERO);
        self.tools.new.centre = [c.x as f32, c.y as f32, c.z as f32];
    }

    /// What a stored value of the displayed series has to be multiplied by
    /// to become a body-weight SUV; `None` for anything that is not a PET
    /// series carrying the header fields it takes.
    pub(super) fn suv_factor(&self, slot: usize) -> Option<f64> {
        let study = self.slots[slot].study.as_ref()?;
        study.series.get(study.active_series)?.suv_bw
    }

    // -- the panel section --------------------------------------------------

    /// The whole editor: the dataset row, then the three sections.
    pub(super) fn structures_editor_section(&mut self, ui: &mut egui::Ui) {
        let title = egui::RichText::new("Structure editor").strong();
        if !self.any_volume() {
            egui::CollapsingHeader::new(title)
                .default_open(true)
                .show(ui, |ui| {
                    ui.weak("Load a dataset with an image volume to draw on");
                });
            ui.separator();
            return;
        }
        self.tools.visible = true;
        // The editor works on one dataset; a dataset that lost its volume
        // hands over to the one that has one.
        if !self.slots[self.tools.slot].has_volume() {
            self.tools.slot = self.first_volume_slot();
        }
        let reveal = self.tools.reveal.take();
        let mut new_slot = None;
        egui::CollapsingHeader::new(title)
            .default_open(true)
            .open(reveal.map(|_| true))
            .show(ui, |ui| {
                new_slot = seg_engines::dataset_row(ui, self.tools.slot, self.volume_slots(), true);
                let open = |s: Section| (reveal == Some(s)).then_some(true);
                egui::CollapsingHeader::new("Insert structure")
                    .default_open(false)
                    .open(open(Section::Insert))
                    .show(ui, |ui| self.insert_section(ui));
                egui::CollapsingHeader::new("Edit structure")
                    .default_open(true)
                    .open(open(Section::Edit))
                    .show(ui, |ui| self.edit_section(ui));
                egui::CollapsingHeader::new("Combine structures")
                    .default_open(false)
                    .open(open(Section::Combine))
                    .show(ui, |ui| self.combine_section(ui));
            });
        ui.separator();
        if let Some(s) = new_slot {
            self.tools.slot = s;
            self.interp = None;
            self.tools.new.status = None;
            self.seed_shape_centre(s);
            self.combine_switch_slot(s);
        }
    }

    // -- the toolbar strip --------------------------------------------------

    /// The nine tools as one row of glyphs, then the options of the tool in
    /// hand, all on the toolbar: what *✏ Draw structure* unfolds.
    pub(super) fn draw_strip(&mut self, ui: &mut egui::Ui) {
        let slot = self.preferred_volume_slot();
        let mut pick: Option<SegTool> = None;
        for (tool, glyph, name, tip, _) in TOOLS {
            if *tool == SegTool::Polygon {
                ui.separator();
            }
            if glyph_button(ui, self.seg_tool == *tool, glyph, &format!("{name}\n{tip}")) {
                pick = Some(*tool);
            }
        }
        if let Some(tool) = pick {
            self.set_seg_tool(if self.seg_tool == tool {
                SegTool::None
            } else {
                tool
            });
        }
        let Some((_, _, name, ..)) = TOOLS.iter().find(|t| t.0 == self.seg_tool) else {
            return;
        };
        ui.separator();
        ui.label(egui::RichText::new(*name).strong());
        if matches!(
            self.seg_tool,
            SegTool::Brush | SegTool::Erase | SegTool::Nudge | SegTool::ContourBrush
        ) {
            ui.add(
                egui::DragValue::new(&mut self.brush_radius_mm)
                    .speed(0.5)
                    .range(0.5..=80.0)
                    .suffix(" mm"),
            )
            .on_hover_text("Brush radius: Shift+wheel, or [ and ]");
        }
        if matches!(self.seg_tool, SegTool::Brush | SegTool::Erase)
            && ui
                .selectable_label(self.brush_3d, "3D")
                .on_hover_text(
                    "Spherical 3D brush: paints through neighbouring slices.\n\
                     Off: flat 2D circle on the displayed slice only",
                )
                .clicked()
        {
            self.brush_3d = !self.brush_3d;
        }
        match self.seg_tool {
            SegTool::Grow => self.grow_options(ui, slot),
            SegTool::ContourBrush => self.smart_brush_options(ui),
            SegTool::LiveWire => self.livewire_options(ui),
            _ => {}
        }
        if self.seg_tool.draws_contours() {
            self.contour_target(ui, slot);
        } else {
            self.segment_target(ui, slot);
        }
    }

    /// Put a tool in hand, or none; whatever the last one left behind goes.
    pub(super) fn set_seg_tool(&mut self, tool: SegTool) {
        self.seg_tool = tool;
        if tool != SegTool::Grow {
            self.cancel_grow();
        }
        if !tool.draws_contours() {
            self.draw = None;
        }
    }

    /// The limiting structure of ✨ Grow: where the front may not go,
    /// whatever the picture says. A box made with *Insert structure* is the
    /// "limiting box" by another name.
    fn grow_options(&mut self, ui: &mut egui::Ui, slot: usize) {
        let cands = self.combine_candidates(slot);
        ui.label("inside:");
        item_picker(
            ui,
            "grow_limit",
            &mut self.grow_limit,
            &cands,
            "no limit",
            160.0,
        )
        .on_hover_text(
            "Keep the region inside this structure, whatever the grey levels do. A box \
             made with Insert structure is the limiting box.",
        );
    }

    /// The smart brush: which tissue the stamp may cover, and how far past
    /// that threshold it may still reach.
    fn smart_brush_options(&mut self, ui: &mut egui::Ui) {
        ui.label("edge:");
        for b in EdgeBand::ALL {
            if ui
                .add(egui::Button::selectable(self.brush_band == b, b.label()).small())
                .on_hover_text(b.hint())
                .clicked()
            {
                self.brush_band = b;
            }
        }
        if self.brush_band != EdgeBand::None {
            ui.add(
                egui::Slider::new(&mut self.brush_sensitivity, 0.0..=1.0)
                    .text("reach")
                    .fixed_decimals(2),
            )
            .on_hover_text(
                "How far past the threshold the brush may still paint, as a fraction of \
                 the display window. Turn it up when the brush stops short of the boundary.",
            );
        }
    }

    /// Training: the live wire learns what the accepted edges look like, so
    /// it prefers that kind of edge over an equally strong one beside it.
    fn livewire_options(&mut self, ui: &mut egui::Ui) {
        if ui
            .checkbox(&mut self.wire_learn, "learn")
            .on_hover_text(
                "Learn from every accepted segment: an edge that looks like the ones \
                 already taken becomes cheaper than an equally strong edge that does not",
            )
            .changed()
        {
            if let Some(w) = &mut self.wire {
                w.training = self.wire_learn;
            }
        }
        if self.livewire_trained()
            && small_tip_button(
                ui,
                "forget",
                "Start again from the plain gradient cost - what to press when moving \
                 from one organ to a different one",
            )
        {
            self.livewire_untrain();
        }
    }

    /// Which structure the contour strokes land in, and what they do to
    /// what is already there.
    fn contour_target(&mut self, ui: &mut egui::Ui, slot: usize) {
        match self.edit_roi_name(slot) {
            Some((name, c)) => {
                ui.label(egui::RichText::new(format!("▸ {name}")).color(theme::rgb(c)))
                    .on_hover_text(
                        "The structure being edited. Select another in the RT structures \
                         list, or Ctrl-click a contour in a view",
                    );
            }
            None => {
                ui.label(egui::RichText::new("▸ new structure").weak())
                    .on_hover_text(
                        "There is no structure to edit yet - the first stroke creates one",
                    );
            }
        }
        if small_tip_button(ui, "+", "New RT structure, and edit it") {
            self.new_roi(slot, None, "ORGAN");
        }
        for m in DrawMode::ALL {
            if ui
                .add(egui::Button::selectable(self.draw_mode == m, m.label()).small())
                .on_hover_text(m.hint())
                .clicked()
            {
                self.draw_mode = m;
            }
        }
    }

    /// Which segmentation the voxel tools paint into.
    fn segment_target(&mut self, ui: &mut egui::Ui, slot: usize) {
        let s = &self.slots[slot];
        match s.segs().get(s.active_seg) {
            Some(seg) => {
                ui.label(
                    egui::RichText::new(format!("▸ {}", seg.name)).color(theme::rgb(seg.color)),
                )
                .on_hover_text(
                    "The segmentation being painted. Select another in the \
                         Segmentations list",
                );
            }
            None => {
                ui.label(egui::RichText::new("▸ new segmentation").weak())
                    .on_hover_text("There is no segmentation yet - the first stroke creates one");
            }
        }
        if small_tip_button(ui, "+", "New, empty segmentation, and paint into it") {
            self.create_seg(slot);
        }
    }

    // -- Edit ---------------------------------------------------------------

    /// Everything that acts on the whole edited structure.
    fn edit_section(&mut self, ui: &mut egui::Ui) {
        let slot = self.tools.slot;
        // The preview is recomputed here, once a frame, and only while it is
        // wanted: it is the one thing in this section that costs anything.
        if self.tools.show_interp {
            self.refresh_interp(slot);
        } else if self.interp.is_some() {
            self.interp = None;
        }
        self.refresh_derived(slot);
        let Some((name, roi_type, vol, occupied, points)) = self.edit_summary(slot) else {
            ui.weak(
                "No structure is selected. Click one in the RT structures list, \
                 Ctrl-click a contour in a view, or just start drawing.",
            );
            return;
        };
        let target = self.edit_target(slot);
        let derived = target
            .and_then(|(set, roi)| self.derived_of(slot, set, roi))
            .map(|d| d.line(&name))
            .zip(target.and_then(|(_, roi)| self.derived_status(slot, roi)));
        let derived_busy = self.derived_job.is_some();
        let axis = self.edit_axis(slot);
        let level = self.edit_level(slot);
        let n_interp = self.interp_count(slot);
        let has_clip = self.contour_clip.is_some();
        let slices = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.volume.dims[axis])
            .unwrap_or(1);
        let mode = self.draw_mode.label().to_lowercase();
        let mut snap = self.interp_snap;
        let mut act: Option<Act> = None;
        let t = &mut self.tools;

        ui.label(egui::RichText::new(format!("✏ {name}")).strong());
        ui.label(
            egui::RichText::new(format!(
                "{vol:.1} cm³ · {occupied} slice(s) · {points} points · {} slices",
                axis_name(axis)
            ))
            .weak(),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Type:");
            egui::ComboBox::from_id_salt("roi_type")
                .selected_text(if roi_type.is_empty() {
                    "(none)"
                } else {
                    roi_type.as_str()
                })
                .show_ui(ui, |ui| {
                    for ty in ROI_TYPES {
                        if ui.selectable_label(roi_type == *ty, *ty).clicked() {
                            act = Some(Act::Retype(ty));
                        }
                    }
                })
                .response
                .on_hover_text("The RT ROI Interpreted Type - what a planning system branches on");
        });

        // -- derived ---------------------------------------------------------
        if let Some((line, st)) = &derived {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Derived").strong());
                ui.label(
                    egui::RichText::new(format!("{} {}", st.glyph(), st.label()))
                        .color(theme::rgb(st.color())),
                );
                act_row(
                    ui,
                    &mut act,
                    &[
                        (
                            "Update",
                            "Run the recipe again on the current operands",
                            !derived_busy,
                            Act::DerivedUpdate,
                        ),
                        (
                            "Edit recipe",
                            "Open it in the Combine window",
                            true,
                            Act::DerivedEdit,
                        ),
                        (
                            "Underive",
                            "Forget the recipe and keep the geometry: an ordinary structure \
                             from here on",
                            true,
                            Act::Underive,
                        ),
                    ],
                );
            });
            ui.label(egui::RichText::new(line.clone()).italics().small());
        }

        // -- interpolation ---------------------------------------------------
        ui.label(egui::RichText::new("Interpolation").strong())
            .on_hover_text(
                "Contours for the slices between the drawn ones, blended through their \
                 distance fields. Shown dashed until accepted - nothing is stored before \
                 that.",
            );
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut t.show_interp, "Show");
            if ui
                .checkbox(&mut snap, "Snap to edges")
                .on_hover_text(
                    "Pull every interpolated contour onto the boundary the image shows \
                     there. The preview is what gets stored, so what is dashed is what \
                     you accept.",
                )
                .changed()
            {
                act = Some(Act::SnapInterp(snap));
            }
            ui.label(format!("{n_interp} slice(s)"));
            let this = format!(
                "Slice {level} of {slices} - the one the {} view shows",
                axis_name(axis)
            );
            act_row(
                ui,
                &mut act,
                &[
                    (
                        "Accept slice",
                        &this,
                        n_interp > 0,
                        Act::AcceptInterp(false),
                    ),
                    ("Accept all", "", n_interp > 0, Act::AcceptInterp(true)),
                ],
            );
        });

        // -- this slice ------------------------------------------------------
        ui.label(egui::RichText::new(format!("Slice {level} of {slices}")).strong())
            .on_hover_text(
                "The slice the view along the drawing plane is showing - the one a \
                 stroke would land on; the wheel moves it",
            );
        ui.horizontal_wrapped(|ui| {
            let paste = format!("Paste them here ({mode} mode)");
            act_row(
                ui,
                &mut act,
                &[
                    ("Copy", "Copy this slice's contours", true, Act::Copy),
                    ("Paste", &paste, has_clip, Act::Paste),
                    (
                        "Delete one",
                        "Delete the single contour the crosshair is inside - the rest of \
                         the slice is left alone",
                        true,
                        Act::DeleteOne,
                    ),
                    (
                        "Clear",
                        "Delete every contour on this slice",
                        true,
                        Act::Clear,
                    ),
                ],
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Thin: keep every");
            ui.add(egui::DragValue::new(&mut t.keep_nth).range(2..=20));
            ui.label("th slice");
            act_row(
                ui,
                &mut act,
                &[(
                    "Apply",
                    "Drop the slices in between, over the whole structure. What \
                     interpolation is for: thin out, correct two slices, interpolate again",
                    true,
                    Act::Thin,
                )],
            );
        });

        // -- tidying ---------------------------------------------------------
        ui.label(egui::RichText::new("Tidy").strong());
        ui.horizontal_wrapped(|ui| {
            act_row(
                ui,
                &mut act,
                &[
                    (
                        "Resolve overlaps",
                        "Contours of this structure that cross each other become one clean \
                         set of nested rings; the filled area does not change",
                        true,
                        Act::ResolveOverlaps,
                    ),
                    (
                        "Remove holes",
                        "Fill every interior cavity, on every slice",
                        true,
                        Act::RemoveHoles,
                    ),
                    (
                        "Keep largest",
                        "Keep the largest contour of each slice, with its holes",
                        true,
                        Act::KeepLargest,
                    ),
                ],
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Drop under");
            ui.add(
                egui::DragValue::new(&mut t.min_area_cm2)
                    .speed(0.01)
                    .range(0.0..=100.0)
                    .suffix(" cm²"),
            );
            act_row(ui, &mut act, &[("Apply", "", true, Act::DropSmall)]);
            ui.label("At most");
            ui.add(egui::DragValue::new(&mut t.max_points).range(8..=2000));
            ui.label("points");
            act_row(
                ui,
                &mut act,
                &[(
                    "Apply",
                    "Simplified with the largest tolerance that still fits the cap, so the \
                     shape is kept and the vertices are the original ones",
                    true,
                    Act::LimitPoints,
                )],
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::DragValue::new(&mut t.smooth).range(1..=10));
            ui.label("smoothing pass(es)");
            act_row(
                ui,
                &mut act,
                &[(
                    "Apply",
                    "Area-preserving: the outline relaxes, the volume stays",
                    true,
                    Act::Smooth,
                )],
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Piece under the crosshair:");
            act_row(
                ui,
                &mut act,
                &[
                    (
                        "Keep it alone",
                        "Keep the connected piece the crosshair is in and drop every other one",
                        true,
                        Act::Component(true),
                    ),
                    (
                        "Delete it",
                        "Delete that piece and keep the rest",
                        true,
                        Act::Component(false),
                    ),
                ],
            );
        });

        // -- transforms ------------------------------------------------------
        ui.label(egui::RichText::new("Move the whole structure").strong())
            .on_hover_text("In the plane it is drawn on");
        ui.horizontal_wrapped(|ui| {
            for (i, prefix) in ["x ", "y "].iter().enumerate() {
                ui.add(
                    egui::DragValue::new(&mut t.shift[i])
                        .speed(0.5)
                        .range(-500.0..=500.0)
                        .prefix(*prefix)
                        .suffix(" mm"),
                );
            }
            act_row(
                ui,
                &mut act,
                &[
                    ("Move", "In millimetres", true, Act::Translate),
                    (
                        "To crosshair",
                        "Put the structure's centroid under the crosshair - the \"move to \
                         slice intersection\" of a planning system",
                        true,
                        Act::ToCrosshair,
                    ),
                ],
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::DragValue::new(&mut t.scale_pct)
                    .speed(0.5)
                    .range(10.0..=400.0)
                    .suffix(" %"),
            );
            act_row(
                ui,
                &mut act,
                &[(
                    "Scale",
                    "About the structure's own centroid",
                    true,
                    Act::Scale,
                )],
            );
            ui.add(
                egui::DragValue::new(&mut t.rotate_deg)
                    .speed(1.0)
                    .range(-180.0..=180.0)
                    .suffix("°"),
            );
            act_row(ui, &mut act, &[("Rotate", "", true, Act::Rotate)]);
        });
        ui.label(
            egui::RichText::new(
                "Every button is one undo step: Ctrl+Z with a contour tool in hand.",
            )
            .weak()
            .small(),
        );
        if let Some(a) = act {
            self.apply_contour_act(slot, a);
        }
    }

    fn apply_contour_act(&mut self, slot: usize, act: Act) {
        let t = &self.tools;
        let (min_area, max_points, smooth, keep, shift, scale, rot) = (
            t.min_area_cm2 as f64,
            t.max_points,
            t.smooth,
            t.keep_nth,
            t.shift,
            t.scale_pct as f64 / 100.0,
            (t.rotate_deg as f64).to_radians(),
        );
        let axis = self.edit_axis(slot);
        let spacing = self.slots[slot].spacing_or_unit();
        let [ua, va] = plane_axes(axis);
        match act {
            Act::AcceptInterp(all) => {
                self.accept_interp(slot, all);
            }
            Act::SnapInterp(on) => {
                self.interp_snap = on;
                // The preview is what gets accepted, so it is rebuilt now
                // rather than at the next edit.
                self.interp = None;
                self.refresh_interp(slot);
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
            Act::Retype(ty) => self.set_edit_roi_type(slot, ty),
        }
    }

    // -- Insert structure ---------------------------------------------------

    /// An empty structure, a point, or a generated structure: source, its
    /// settings, name and type, Create.
    fn insert_section(&mut self, ui: &mut egui::Ui) {
        let slot = self.tools.slot;
        // A shape that was never placed starts at the crosshair.
        if self.tools.new.centre == [0.0; 3] {
            self.seed_shape_centre(slot);
        }
        let (mut add_roi, mut add_poi) = (false, false);
        ui.horizontal_wrapped(|ui| {
            add_roi = small_tip_button(
                ui,
                "+ Empty structure",
                "New, empty structure in the active set - and the one the contour tools \
                 draw into",
            );
            add_poi = small_tip_button(
                ui,
                "✱ Point of interest",
                "A marker, a reference point or the point the patient is lined up on, at \
                 the crosshair. It exports as a POINT contour like any other",
            );
        });
        if add_roi {
            self.new_roi(slot, None, "ORGAN");
        }
        if add_poi && self.new_poi(slot, None).is_none() {
            self.notice = Some("There is no image volume to place a point on.".into());
        }
        ui.label(egui::RichText::new("Generate").strong());
        let candidates = self.combine_candidates(slot);
        let has_dose = self.slots[slot]
            .study
            .as_ref()
            .is_some_and(|s| !s.doses.is_empty());
        let reference = self.slots[slot].dose_reference;
        let suv_factor = self.suv_factor(slot);
        let mut create = false;
        let mut seed = false;
        let d = &mut self.tools.new;
        ui.horizontal_wrapped(|ui| {
            ui.label("From:");
            for s in Source::ALL {
                if ui
                    .add_enabled(
                        s != Source::Dose || has_dose,
                        egui::Button::selectable(d.source == s, s.label()).small(),
                    )
                    .clicked()
                {
                    d.source = s;
                }
            }
        });
        match d.source {
            Source::GreyLevel => {
                if suv_factor.is_some() {
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .checkbox(&mut d.suv, "SUV (body weight)")
                            .on_hover_text(
                                "Read the two bounds as standardized uptake values instead \
                                 of stored counts. The factor comes from this series' own \
                                 header: patient weight, injected activity and the decay to \
                                 the acquisition time.",
                            )
                            .changed()
                        {
                            // Carry the window across the change of units
                            // instead of leaving numbers that mean something
                            // entirely different.
                            let f = suv_factor.unwrap_or(1.0);
                            let (lo, hi) = if d.suv {
                                (d.lo as f64 * f, d.hi as f64 * f)
                            } else {
                                (d.lo as f64 / f, d.hi as f64 / f)
                            };
                            d.lo = lo as f32;
                            d.hi = hi as f32;
                        }
                        if d.suv {
                            ui.weak("2.5 is the usual start for a lesion");
                        }
                    });
                }
                let unit = if d.suv { "" } else { " HU" };
                let range = if d.suv { 0.0..=100.0 } else { -2000.0..=6000.0 };
                let step = if d.suv { 0.1 } else { 5.0 };
                ui.horizontal_wrapped(|ui| {
                    ui.label("Between");
                    ui.add(
                        egui::DragValue::new(&mut d.lo)
                            .speed(step)
                            .range(range.clone())
                            .suffix(unit),
                    );
                    ui.label("and");
                    ui.add(
                        egui::DragValue::new(&mut d.hi)
                            .speed(step)
                            .range(range)
                            .suffix(unit),
                    );
                    if !d.suv {
                        ui.label("bone:");
                        for (label, lo) in generate::BONE_PRESETS {
                            if small_tip_button(ui, label, format!("Everything above {lo:.0} HU")) {
                                d.lo = lo;
                                d.hi = 6000.0;
                            }
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Only inside:");
                    item_picker(
                        ui,
                        "newroi_limit",
                        &mut d.limit,
                        &candidates,
                        "(the whole image)",
                        200.0,
                    );
                });
            }
            Source::Shape => {
                ui.horizontal_wrapped(|ui| {
                    for s in Shape::ALL {
                        if ui
                            .add(egui::Button::selectable(d.shape == s, s.label()).small())
                            .clicked()
                        {
                            d.shape = s;
                            if s.is_uniform() {
                                d.half = [d.half[0]; 3];
                            }
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    if d.shape.is_uniform() {
                        ui.label("Radius:");
                        let mut r = d.half[0];
                        if ui
                            .add(
                                egui::DragValue::new(&mut r)
                                    .speed(0.5)
                                    .range(0.5..=400.0)
                                    .suffix(" mm"),
                            )
                            .changed()
                        {
                            d.half = [r; 3];
                        }
                    } else {
                        ui.label("Half-sizes x/y/z:");
                        for v in &mut d.half {
                            ui.add(
                                egui::DragValue::new(v)
                                    .speed(0.5)
                                    .range(0.5..=400.0)
                                    .suffix(" mm"),
                            );
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Centre:");
                    for v in &mut d.centre {
                        ui.add(egui::DragValue::new(v).speed(1.0).suffix(" mm"));
                    }
                    if small_tip_button(ui, "⌖", "Take the centre from the crosshair") {
                        seed = true;
                    }
                })
                .response
                .on_hover_text("Patient coordinates");
            }
            Source::Dose => {
                ui.horizontal_wrapped(|ui| {
                    ui.label("At least");
                    if d.dose_absolute {
                        ui.add(
                            egui::DragValue::new(&mut d.dose_gy)
                                .speed(0.5)
                                .range(0.0..=1000.0)
                                .suffix(" Gy"),
                        );
                    } else {
                        ui.add(
                            egui::DragValue::new(&mut d.dose_pct)
                                .speed(1.0)
                                .range(0.0..=200.0)
                                .suffix(" %"),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "of {reference:.1} Gy = {:.1} Gy",
                                reference * d.dose_pct / 100.0
                            ))
                            .weak(),
                        );
                    }
                    ui.checkbox(&mut d.dose_absolute, "absolute").on_hover_text(
                        "The displayed dose, sampled the way the isodose lines are, so the \
                         structure agrees with what is on screen",
                    );
                });
            }
            Source::Fov => {
                ui.label(
                    egui::RichText::new(
                        "The part of the images that carries data: the circle a CT was \
                         reconstructed on, found from the value its corners are padded \
                         with. A threshold or a body contour limited to it stops at the \
                         edge of the data instead of the edge of the matrix.",
                    )
                    .weak(),
                );
            }
        }
        if d.source != Source::Shape {
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut d.keep_largest, "keep the largest piece");
                ui.checkbox(&mut d.fill_holes, "fill cavities");
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::TextEdit::singleline(&mut d.name).desired_width(120.0))
                .on_hover_text("Name of the new structure");
            egui::ComboBox::from_id_salt("newroi_type")
                .selected_text(&d.roi_type)
                .width(100.0)
                .show_ui(ui, |ui| {
                    for ty in &ROI_TYPES[..6] {
                        ui.selectable_value(&mut d.roi_type, ty.to_string(), *ty);
                    }
                });
            if tip_button(ui, "▶ Create", "On the displayed image series") {
                create = true;
            }
        });
        if let Some(status) = &d.status {
            ui.weak(status);
        }
        if seed {
            self.seed_shape_centre(slot);
        }
        if create {
            self.create_newroi(slot);
        }
    }

    /// Build the mask the current settings describe.
    fn newroi_mask(&self, slot: usize) -> Result<Vec<u8>, String> {
        let d = &self.tools.new;
        let study = self.slots[slot]
            .study
            .as_ref()
            .ok_or_else(|| "no dataset".to_string())?;
        let grid = study.volume.grid();
        match d.source {
            Source::GreyLevel => {
                let limit = match d.limit {
                    Some(item) => Some(self.operand_mask(slot, item, &grid).ok_or_else(|| {
                        "the limiting structure has nothing on this image series".to_string()
                    })?),
                    None => None,
                };
                // The threshold is always applied in the stored values the
                // volume holds; an SUV window is converted back into them,
                // which is exact because the SUV is a plain scale factor.
                let (lo, hi) = match self.suv_factor(slot).filter(|_| d.suv) {
                    Some(f) => ((d.lo as f64 / f) as f32, (d.hi as f64 / f) as f32),
                    None => (d.lo, d.hi),
                };
                Ok(generate::threshold_mask(
                    &study.volume,
                    lo,
                    hi,
                    limit.as_deref(),
                ))
            }
            Source::Fov => generate::fov_mask(&study.volume).ok_or_else(|| {
                "These images are not reconstructed on a smaller field of view - their \
                 corners hold data like the rest, so there is nothing to outline."
                    .to_string()
            }),
            Source::Shape => Ok(generate::shape_mask(
                d.shape,
                &grid,
                Vec3::new(d.centre[0] as f64, d.centre[1] as f64, d.centre[2] as f64),
                [d.half[0] as f64, d.half[1] as f64, d.half[2] as f64],
            )),
            Source::Dose => {
                let dose = study
                    .doses
                    .get(self.slots[slot].active_dose)
                    .ok_or_else(|| "this dataset has no dose".to_string())?;
                let level = if d.dose_absolute {
                    d.dose_gy
                } else {
                    self.slots[slot].dose_reference * d.dose_pct / 100.0
                };
                Ok(generate::dose_mask(&grid, dose, level))
            }
        }
    }

    /// Make the structure, and say what came out.
    fn create_newroi(&mut self, slot: usize) {
        let mut mask = match self.newroi_mask(slot) {
            Ok(m) => m,
            Err(why) => {
                self.error = Some(format!("Nothing was created: {why}."));
                return;
            }
        };
        let Some(grid) = self.slots[slot].study.as_ref().map(|s| s.volume.grid()) else {
            return;
        };
        let (source, keep_largest, fill_holes) = {
            let d = &self.tools.new;
            (d.source, d.keep_largest, d.fill_holes)
        };
        // The two tidying options every generator wants, and no more: a
        // threshold picks up specks, and a bone ROI is full of marrow.
        if source != Source::Shape && (keep_largest || fill_holes) {
            let cleanup = crate::structops::Cleanup {
                fill_holes,
                keep_largest,
                ..crate::structops::Cleanup::default()
            };
            cleanup.apply(&mut mask, &grid, &crate::progress::Quiet);
        }
        if crate::morphology::count_set(&mask) == 0 {
            self.error = Some(
                "Nothing was created: no voxel of the displayed series matches. Widen \
                 the window, or check the limiting structure."
                    .into(),
            );
            return;
        }
        let stack = Stack::from_mask(&mask, grid.dims, 2);
        let cm3 = stack.volume_cm3(grid.spacing);
        let (name, roi_type) = (self.tools.new.name.clone(), self.tools.new.roi_type.clone());
        let Some(roi) = self.new_roi(slot, Some(name.clone()), &roi_type) else {
            return;
        };
        let set_idx = self.slots[slot].active_structs;
        if let Some(r) = self.slots[slot].roi_mut(set_idx, roi) {
            stack.apply_to_roi(r, &grid);
        }
        self.settings_gen += 1;
        self.edit = None;
        self.tools.new.status = Some(format!(
            "✔ {name} created: {cm3:.1} cm³ on {} slice(s). It is now the structure the \
             editor works on.",
            stack.occupied()
        ));
    }
}

/// The nine tools of the *Draw* row: the tool, its glyph, its name, the
/// tooltip that explains it, and the one-line mouse summary the status bar
/// shows.
const TOOLS: &[(SegTool, &str, &str, &str, &str)] = &[
    (
        SegTool::Brush,
        "🎨",
        "Paint",
        "Paint the active segmentation (LMB drag).\n\
         Hold Alt to erase · Shift+wheel or [ ] resize the brush · Ctrl+Z undo",
        "LMB paint · Alt erase · Shift+wheel / [ ] brush size · Ctrl+Z undo · wheel slice"
    ),
    (
        SegTool::Erase,
        "⊖",
        "Erase",
        "Erase from the active segmentation (LMB drag)",
        "LMB erase · Shift+wheel / [ ] brush size · Ctrl+Z undo · wheel slice"
    ),
    (
        SegTool::Grow,
        "✨",
        "Grow",
        "Interactive organ segmentation (geodesic fast marching): press to place a seed, \
         drag up/down to grow/shrink the region with a live preview. Intensity changes \
         and edges act as barriers, so the organ under the seed is suggested before \
         anything leaks. Release commits (enclosed holes are filled), Esc cancels",
        "LMB press seed · drag up/down = grow/shrink · release commit · Esc cancel · Ctrl+Z undo"
    ),
    (
        SegTool::Polygon,
        "📐",
        "Polygon",
        "Draw a contour into the active RT structure, click by click (right-click, \
         double-click or Enter closes it, Esc cancels).\n\
         Ctrl-click picks the structure under the pointer · Ctrl+Z undo",
        "LMB add point · RMB / double-click / Enter close · Esc cancel · Ctrl-click pick structure · Ctrl+Z undo"
    ),
    (
        SegTool::Spline,
        "✒",
        "Spline",
        "The same, but the clicked points are joined by a closed spline - four clicks \
         for a smooth organ outline",
        "LMB add point · RMB / double-click / Enter close · Esc cancel · Ctrl-click pick structure · Ctrl+Z undo"
    ),
    (
        SegTool::Freehand,
        "✏",
        "Free",
        "Draw a contour freehand: press, drag round the structure, release. The stroke \
         is closed and thinned on release",
        "LMB drag draws · release closes · Ctrl-click pick structure · Ctrl+Z undo"
    ),
    (
        SegTool::LiveWire,
        "🔗",
        "Live wire",
        "Draw along the edge under the pointer: click once on the boundary, move along \
         it, and the curve between the two follows the image gradient instead of the \
         straight line. Click to anchor what is on screen; right-click, double-click or \
         Enter closes, Esc cancels",
        "LMB anchors the path along the edge · RMB / double-click / Enter close · Esc cancel · Ctrl-click pick structure · Ctrl+Z undo"
    ),
    (
        SegTool::ContourBrush,
        "🖊",
        "Brush",
        "Paint into the active RT structure: a round brush on the patient that pushes \
         the contour lines (LMB drag).\n\
         Hold Alt to erase · Shift+wheel or [ ] resize · Ctrl+Z undo",
        "LMB paints the structure · Alt erases · Shift+wheel / [ ] radius · Ctrl+Z undo"
    ),
    (
        SegTool::Nudge,
        "⌖",
        "Nudge",
        "Push the outline of the edited structure around: vertices within the tool \
         radius follow the drag, with a smooth falloff. Shift+wheel or [ ] set the radius",
        "LMB drag pushes the outline · Shift+wheel / [ ] radius · Ctrl+Z undo"
    ),
];

/// What the mouse does with `tool` in hand - the status bar's one line.
pub(super) fn mouse_hint(tool: SegTool) -> &'static str {
    TOOLS
        .iter()
        .find(|t| t.0 == tool)
        .map(|t| t.4)
        .unwrap_or("LMB crosshair · RMB W/L · MMB pan · wheel slice · Ctrl+wheel zoom")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_is_in_the_row_once_with_its_own_glyph() {
        let mut glyphs: Vec<&str> = TOOLS.iter().map(|t| t.1).collect();
        glyphs.sort();
        glyphs.dedup();
        assert_eq!(glyphs.len(), TOOLS.len());
        assert_eq!(TOOLS.len(), 9);
        assert!(TOOLS.iter().all(|t| t.0 != SegTool::None));
        assert!(mouse_hint(SegTool::ContourBrush).starts_with("LMB paints"));
        assert!(mouse_hint(SegTool::None).starts_with("LMB crosshair"));
    }

    #[test]
    fn the_generator_starts_from_a_bone_window_and_a_sphere() {
        let n = NewRoi::default();
        assert!(n.lo < n.hi);
        assert_eq!(n.shape, Shape::Sphere);
        assert!(!n.suv);
        assert!(ROI_TYPES.contains(&n.roi_type.as_str()));
    }
}
