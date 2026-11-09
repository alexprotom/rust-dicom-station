//! The contour tools of the MPR views on a planar pane: polygon, spline,
//! freehand, live wire, contour brush and nudge, with the Auto / Extend /
//! Subtract rule and undo, on the 2D contours of a planar structure set.
//!
//! The pane is a second *target* of the toolbar's Draw structure row: a
//! stroke on a planar pane makes it the target (`Planar2d::target`), a
//! stroke in an MPR view makes the volume the target again, and the row,
//! Ctrl+Z, Esc and Enter follow the target. A stroke is collected in the
//! image's pixels (centre of the first pixel at `(0, 0)`), made a ring the
//! way `commit_draw` makes one, and applied straight to the structure's
//! [`crate::contours::Region`] on that image - the region is the stored
//! form, so there is no working stack to flush. The rule, the spline and
//! the brush stamp are the volume tools' own ([`contours::auto_cuts`],
//! [`contours::spline_points`], [`crate::planar_structs::brush`]); the
//! brush radius is millimetres of the image's own pixel spacing. The live
//! wire runs on the image's pixels under the pane's window. The voxel tools
//! (brush, erase, grow) have no planar meaning and do nothing on a pane.

use super::*;

use crate::contours::{self, Poly};
use crate::planar_structs::{ContourSource, PlanarContour};

/// How deep the planar undo goes.
const PLANAR_UNDO: usize = 32;

/// A contour being drawn on one planar image, in its pixels.
pub(super) struct PlanarDraw {
    pub(super) slot: usize,
    pub(super) uid: String,
    pub(super) pts: Vec<[f64; 2]>,
}

/// One planar structure's contours before an edit.
pub(super) struct PlanarSnapshot {
    slot: usize,
    set: usize,
    structure: String,
    contours: Vec<PlanarContour>,
}

/// The live wire's costs over one planar image under one window.
pub(super) struct PlanarWire {
    slot: usize,
    uid: String,
    window: (f32, f32),
    costs: crate::livewire::Costs,
    tree: Option<crate::livewire::Tree>,
}

/// What the pointer did on a planar pane with a contour tool in hand,
/// applied by [`ViewerApp::planar_contour_apply`].
pub(super) enum PlanarEdit {
    /// A click of the polygon, spline or live wire; the freehand's press.
    Click(usize, [f64; 2]),
    /// The freehand drag's samples.
    Drag([f64; 2]),
    /// Close the contour under construction.
    Close,
    /// A sample of the contour brush; `erase` with Alt.
    Brush {
        image: usize,
        at: [f64; 2],
        erase: bool,
    },
    /// A sample of the nudge.
    Nudge { image: usize, at: [f64; 2] },
    /// The end of a brush or nudge stroke.
    StrokeEnd,
    /// Ctrl-click: make the structure under the pointer the edited one.
    Pick(usize, [f64; 2]),
}

impl ViewerApp {
    /// The workspace whose planar pane is the contour tools' target, while a
    /// contour tool is in hand.
    pub(super) fn planar_tool_target(&self) -> Option<usize> {
        self.planar2d
            .target
            .filter(|_| self.seg_tool.draws_contours())
    }

    /// The pointer on a planar pane with a contour tool in hand, over image
    /// `image` (`dims` its raster) drawn in `on`: the gestures of the MPR
    /// views, in the image's pixels.
    pub(super) fn planar_contour_input(
        &self,
        ui: &egui::Ui,
        image: usize,
        dims: [usize; 2],
        on: Rect,
        resp: &egui::Response,
    ) -> Vec<PlanarEdit> {
        let mut out = Vec::new();
        let at = |mp: Pos2| super::planar_contours::screen_to_px(mp, dims, on);
        let ctrl = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
        let alt = ui.input(|i| i.modifiers.alt);
        let pos = resp.interact_pointer_pos();
        match self.seg_tool {
            // Click by click, closed with the right button, Enter or a
            // double click; Ctrl-click picks the structure under the pointer.
            SegTool::Polygon | SegTool::Spline | SegTool::LiveWire => {
                if resp.clicked() {
                    if let Some(mp) = pos {
                        out.push(if ctrl {
                            PlanarEdit::Pick(image, at(mp))
                        } else {
                            PlanarEdit::Click(image, at(mp))
                        });
                    }
                }
                if resp.clicked_by(egui::PointerButton::Secondary) || resp.double_clicked() {
                    out.push(PlanarEdit::Close);
                }
            }
            // One drag, one contour.
            SegTool::Freehand => {
                if ctrl && resp.clicked() {
                    if let Some(mp) = pos {
                        out.push(PlanarEdit::Pick(image, at(mp)));
                    }
                } else if resp.drag_started_by(egui::PointerButton::Primary) {
                    if let Some(mp) = pos {
                        out.push(PlanarEdit::Click(image, at(mp)));
                    }
                } else if resp.dragged_by(egui::PointerButton::Primary) {
                    if let Some(mp) = pos {
                        out.push(PlanarEdit::Drag(at(mp)));
                    }
                }
                if resp.drag_stopped_by(egui::PointerButton::Primary) {
                    out.push(PlanarEdit::Close);
                }
            }
            SegTool::ContourBrush => {
                if resp.dragged_by(egui::PointerButton::Primary) || resp.clicked() {
                    if let Some(mp) = pos {
                        out.push(PlanarEdit::Brush {
                            image,
                            at: at(mp),
                            erase: alt,
                        });
                    }
                }
                if resp.drag_stopped_by(egui::PointerButton::Primary) || resp.clicked() {
                    out.push(PlanarEdit::StrokeEnd);
                }
            }
            SegTool::Nudge => {
                if resp.dragged_by(egui::PointerButton::Primary) {
                    if let Some(mp) = pos {
                        out.push(PlanarEdit::Nudge { image, at: at(mp) });
                    }
                }
                if resp.drag_stopped_by(egui::PointerButton::Primary) || resp.clicked() {
                    out.push(PlanarEdit::StrokeEnd);
                }
            }
            _ => {}
        }
        out
    }

    /// The (set, structure) the planar tools edit on `image` of `slot`: the
    /// active one when its set is unlocked, else a new structure in an
    /// unlocked set that has contours on that image, else in a new set
    /// labelled after the image.
    pub(super) fn ensure_planar_target(
        &mut self,
        slot: usize,
        image: usize,
    ) -> Option<(usize, usize)> {
        let active = self.slots[slot].active_planar;
        let study = self.slots[slot].study.as_mut()?;
        let img = study.planar_images.get(image)?;
        let uid = img.sop_instance_uid.clone();
        if uid.is_empty() {
            return None;
        }
        if let Some((k, j)) = active {
            if study
                .planar_structs
                .get(k)
                .is_some_and(|s| !s.locked && j < s.structures.len())
            {
                return Some((k, j));
            }
        }
        let k = match study
            .planar_structs
            .iter()
            .position(|s| !s.locked && s.images.contains(&uid))
        {
            Some(k) => k,
            None => {
                let label = img.label.clone();
                let study_uid = img.study_uid.clone();
                study
                    .planar_structs
                    .push(crate::planar_structs::PlanarStructureSet::new(
                        &label, &study_uid, "drawn",
                    ));
                study.planar_structs.len() - 1
            }
        };
        let set = &mut study.planar_structs[k];
        let mut n = set.structures.len() + 1;
        while set.structure(&format!("2D {n}")).is_some() {
            n += 1;
        }
        set.structure_entry(&format!("2D {n}"), None);
        let j = set.structures.len() - 1;
        self.slots[slot].active_planar = Some((k, j));
        self.settings_gen += 1;
        Some((k, j))
    }

    /// `+` of the toolbar on a planar target: a new structure, in the set of
    /// the image being drawn on (or the first one a pane shows).
    pub(super) fn new_planar_structure(&mut self, slot: usize) {
        let uid = self
            .planar2d
            .draw
            .as_ref()
            .map(|d| d.uid.clone())
            .or_else(|| self.planar_shown_uid(slot));
        let Some(image) = uid.and_then(|u| {
            self.slots[slot]
                .study
                .as_ref()
                .and_then(|st| st.planar_by_uid(&u))
        }) else {
            return;
        };
        self.slots[slot].active_planar = None;
        self.ensure_planar_target(slot, image);
    }

    /// The UID of the first image a planar pane of `slot` shows.
    fn planar_shown_uid(&self, slot: usize) -> Option<String> {
        let study = self.slots[slot].study.as_ref()?;
        (0..settings::PLANAR_PANES).find_map(|k| {
            let l = self.slots[slot].planar[k].layers.first()?;
            study
                .planar_images
                .get(l.image)
                .map(|img| img.sop_instance_uid.clone())
        })
    }

    /// The set label, name and colour of the structure the planar tools
    /// edit.
    pub(super) fn planar_target_name(&self, slot: usize) -> Option<(String, String, [u8; 3])> {
        let (k, j) = self.slots[slot].active_planar?;
        let set = self.slots[slot].study.as_ref()?.planar_structs.get(k)?;
        let s = set.structures.get(j)?;
        Some((set.label.clone(), s.name.clone(), s.color))
    }

    fn push_planar_undo(&mut self, slot: usize, k: usize, j: usize) {
        let Some(s) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_structs.get(k))
            .and_then(|set| set.structures.get(j))
        else {
            return;
        };
        self.planar2d.undo.push(PlanarSnapshot {
            slot,
            set: k,
            structure: s.name.clone(),
            contours: s.contours.clone(),
        });
        if self.planar2d.undo.len() > PLANAR_UNDO {
            self.planar2d.undo.remove(0);
        }
    }

    /// Ctrl+Z on a planar target: the last edit of a planar structure of
    /// `slot` taken back.
    pub(super) fn undo_planar_edit(&mut self, slot: usize) {
        let Some(pos) = self.planar2d.undo.iter().rposition(|u| u.slot == slot) else {
            return;
        };
        let u = self.planar2d.undo.remove(pos);
        if let Some(set) = self.slots[slot]
            .study
            .as_mut()
            .and_then(|st| st.planar_structs.get_mut(u.set))
        {
            if let Some(s) = set.structures.iter_mut().find(|s| s.name == u.structure) {
                s.contours = u.contours;
            }
            set.refresh_images();
        }
        self.settings_gen += 1;
    }

    /// The SOP UID, pixel spacing and raster of planar image `image`.
    fn planar_image_of(&self, slot: usize, image: usize) -> Option<(String, [f64; 2], [usize; 2])> {
        let img = self.slots[slot].study.as_ref()?.planar_images.get(image)?;
        Some((
            img.sop_instance_uid.clone(),
            img.spacing,
            [img.cols, img.rows],
        ))
    }

    /// Apply what the pointer did on a pane of `slot`; `wl` is the window
    /// the image is shown with (the live wire's picture).
    pub(super) fn planar_contour_apply(
        &mut self,
        slot: usize,
        edits: Vec<PlanarEdit>,
        wl: (f32, f32),
    ) {
        for e in edits {
            self.planar2d.target = Some(slot);
            match e {
                PlanarEdit::Click(image, p) => {
                    let Some((uid, ..)) = self.planar_image_of(slot, image) else {
                        continue;
                    };
                    let restart = !matches!(&self.planar2d.draw,
                        Some(d) if d.slot == slot && d.uid == uid);
                    if restart {
                        self.planar2d.draw = Some(PlanarDraw {
                            slot,
                            uid,
                            pts: Vec::new(),
                        });
                        if let Some(w) = &mut self.planar2d.wire {
                            w.tree = None;
                        }
                    }
                    if self.seg_tool == SegTool::LiveWire {
                        self.planar_wire_click(slot, image, p, wl);
                    } else if let Some(d) = &mut self.planar2d.draw {
                        // A click on the last point asks to close, not for a
                        // zero-length edge.
                        if d.pts
                            .last()
                            .is_none_or(|q| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) > 1e-6)
                        {
                            d.pts.push(p);
                        }
                    }
                }
                PlanarEdit::Drag(p) => {
                    if let Some(d) = &mut self.planar2d.draw {
                        d.pts.push(p);
                    }
                }
                PlanarEdit::Close => self.commit_planar_draw(),
                PlanarEdit::Brush { image, at, erase } => self.planar_brush(slot, image, at, erase),
                PlanarEdit::Nudge { image, at } => self.planar_nudge(slot, image, at),
                PlanarEdit::StrokeEnd => {
                    self.planar2d.last_sample = None;
                    self.planar2d.stroke_saved = false;
                }
                PlanarEdit::Pick(image, p) => {
                    let Some((uid, ..)) = self.planar_image_of(slot, image) else {
                        continue;
                    };
                    let hit = self.slots[slot].study.as_ref().and_then(|st| {
                        st.planar_structs.iter().enumerate().find_map(|(k, set)| {
                            set.structures
                                .iter()
                                .position(|s| s.contour(&uid).is_some_and(|c| c.region.contains(p)))
                                .map(|j| (k, j))
                        })
                    });
                    if hit.is_some() {
                        self.slots[slot].active_planar = hit;
                    }
                }
            }
        }
    }

    /// Close the planar contour under construction and apply it in the
    /// current mode.
    pub(super) fn commit_planar_draw(&mut self) {
        if let Some(w) = &mut self.planar2d.wire {
            w.tree = None;
        }
        let Some(d) = self.planar2d.draw.take() else {
            return;
        };
        if d.pts.len() < 3 {
            return;
        }
        let ring = match self.seg_tool {
            SegTool::Spline => Poly::new(contours::spline_points(&d.pts, 12)),
            // A freehand drag arrives with a point per mouse event; the live
            // wire with one per pixel of its path.
            SegTool::Freehand => Poly::new(d.pts).simplify(0.08),
            SegTool::LiveWire => Poly::new(d.pts).simplify(0.1),
            _ => Poly::new(d.pts),
        };
        let Some(image) = self.slots[d.slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_by_uid(&d.uid))
        else {
            return;
        };
        self.planar_apply_ring(d.slot, image, ring);
    }

    /// One closed ring onto the edited structure's contour on `image`, by
    /// the drawing mode's rule.
    pub(super) fn planar_apply_ring(&mut self, slot: usize, image: usize, ring: Poly) {
        if ring.is_empty() {
            return;
        }
        let Some((uid, ..)) = self.planar_image_of(slot, image) else {
            return;
        };
        let Some((k, j)) = self.ensure_planar_target(slot, image) else {
            return;
        };
        self.push_planar_undo(slot, k, j);
        let mode = self.draw_mode;
        let Some(set) = self.slots[slot]
            .study
            .as_mut()
            .and_then(|st| st.planar_structs.get_mut(k))
        else {
            return;
        };
        let s = &mut set.structures[j];
        let c = s.contour_entry(&uid);
        let cut = match mode {
            DrawMode::Extend => false,
            DrawMode::Subtract => true,
            DrawMode::Auto => contours::auto_cuts(&c.region, &ring),
        };
        crate::planar_structs::apply_ring(&mut c.region, &ring, cut);
        c.source = ContourSource::Drawn;
        s.prune();
        set.refresh_images();
        self.settings_gen += 1;
    }

    fn planar_brush(&mut self, slot: usize, image: usize, at: [f64; 2], erase: bool) {
        let Some((uid, spacing, _)) = self.planar_image_of(slot, image) else {
            return;
        };
        let first = self.planar2d.last_sample.is_none();
        let from = self.planar2d.last_sample.unwrap_or(at);
        self.planar2d.last_sample = Some(at);
        let Some((k, j)) = self.ensure_planar_target(slot, image) else {
            return;
        };
        if first {
            self.push_planar_undo(slot, k, j);
        }
        let mode = self.draw_mode;
        let r = self.brush_radius_mm as f64;
        let Some(set) = self.slots[slot]
            .study
            .as_mut()
            .and_then(|st| st.planar_structs.get_mut(k))
        else {
            return;
        };
        let s = &mut set.structures[j];
        let c = s.contour_entry(&uid);
        // Auto: a stroke that starts inside adds, one that starts outside
        // cuts, decided once at the start of the stroke.
        if first && mode == DrawMode::Auto {
            self.planar2d.brush_cut = !c.region.contains(from);
        }
        let cut = erase
            || match mode {
                DrawMode::Subtract => true,
                DrawMode::Extend => false,
                DrawMode::Auto => self.planar2d.brush_cut,
            };
        crate::planar_structs::brush(&mut c.region, from, at, r, spacing, cut);
        c.source = ContourSource::Drawn;
        s.prune();
        set.refresh_images();
        self.settings_gen += 1;
    }

    fn planar_nudge(&mut self, slot: usize, image: usize, at: [f64; 2]) {
        let Some((uid, spacing, _)) = self.planar_image_of(slot, image) else {
            return;
        };
        let Some(from) = self.planar2d.last_sample.replace(at) else {
            return;
        };
        let Some((k, j)) = self.slots[slot].active_planar else {
            return;
        };
        let editable = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_structs.get(k))
            .filter(|set| !set.locked)
            .and_then(|set| set.structures.get(j))
            .and_then(|s| s.contour(&uid))
            .is_some();
        if !editable {
            return;
        }
        // One undo step per stroke, taken before its first moving sample.
        if !self.planar2d.stroke_saved {
            self.push_planar_undo(slot, k, j);
            self.planar2d.stroke_saved = true;
        }
        let r = self.brush_radius_mm as f64;
        if let Some(c) = self.slots[slot]
            .study
            .as_mut()
            .and_then(|st| st.planar_structs.get_mut(k))
            .and_then(|set| set.structures.get_mut(j))
            .and_then(|s| s.contour_mut(&uid))
        {
            if c.region.nudge(from, at, r, spacing) {
                c.source = ContourSource::Drawn;
                self.settings_gen += 1;
            }
        }
    }

    /// One click of the live wire on a planar image: the path from the last
    /// anchor accepted, the next one started here.
    fn planar_wire_click(&mut self, slot: usize, image: usize, p: [f64; 2], wl: (f32, f32)) {
        let Some(img) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_images.get(image))
        else {
            return;
        };
        let uid = img.sop_instance_uid.clone();
        let (w, h) = (img.cols, img.rows);
        if w < 3 || h < 3 || img.data.len() < w * h {
            return;
        }
        let fresh = self
            .planar2d
            .wire
            .as_ref()
            .is_some_and(|x| x.slot == slot && x.uid == uid && x.window == wl);
        if !fresh {
            let costs = crate::livewire::Costs::new(&img.data, w, h, wl);
            self.planar2d.wire = Some(PlanarWire {
                slot,
                uid,
                window: wl,
                costs,
                tree: None,
            });
        }
        let px = [
            p[0].round().clamp(0.0, (w - 1) as f64) as usize,
            p[1].round().clamp(0.0, (h - 1) as f64) as usize,
        ];
        let learn = self.wire_learn;
        let Some(wire) = self.planar2d.wire.as_mut() else {
            return;
        };
        let accepted: Vec<[usize; 2]> = wire
            .tree
            .as_ref()
            .map(|t| t.path_to(px))
            .unwrap_or_default();
        if learn && accepted.len() >= 2 {
            wire.costs.train(&accepted);
        }
        wire.tree = Some(wire.costs.tree(px));
        if let Some(d) = &mut self.planar2d.draw {
            if accepted.len() >= 2 {
                d.pts
                    .extend(accepted[1..].iter().map(|q| [q[0] as f64, q[1] as f64]));
            } else {
                d.pts.push(p);
            }
        }
    }

    /// Esc on a planar target: the contour under construction dropped.
    pub(super) fn drop_planar_draw(&mut self) {
        self.planar2d.draw = None;
        if let Some(w) = &mut self.planar2d.wire {
            w.tree = None;
        }
    }

    /// The contour under construction, the live wire's rubber band and the
    /// brush's circle over image `image` drawn in `on`; `hover` is the
    /// pointer in its pixels.
    pub(super) fn draw_planar_preview(
        &self,
        painter: &egui::Painter,
        slot: usize,
        image: usize,
        on: Rect,
        hover: Option<[f64; 2]>,
    ) {
        if !self.seg_tool.draws_contours() {
            return;
        }
        let Some(img) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_images.get(image))
        else {
            return;
        };
        let dims = [img.cols, img.rows];
        let to_screen = |p: [f64; 2]| super::planar_contours::px_to_screen(p, dims, on);
        let color = self
            .planar_target_name(slot)
            .map(|(_, _, c)| Color32::from_rgb(c[0], c[1], c[2]))
            .unwrap_or(Color32::YELLOW);
        let stroke = egui::Stroke::new(1.5, color);
        if let Some(d) = self
            .planar2d
            .draw
            .as_ref()
            .filter(|d| d.slot == slot && d.uid == img.sop_instance_uid)
        {
            let pts: Vec<[f64; 2]> = if self.seg_tool == SegTool::Spline && d.pts.len() >= 3 {
                contours::spline_points(&d.pts, 12)
            } else {
                d.pts.clone()
            };
            let mut screen: Vec<Pos2> = pts.iter().map(|p| to_screen(*p)).collect();
            if let Some(h) = hover.filter(|_| self.seg_tool != SegTool::Freehand) {
                let tree = self
                    .planar2d
                    .wire
                    .as_ref()
                    .filter(|w| w.uid == img.sop_instance_uid)
                    .and_then(|w| w.tree.as_ref());
                match (self.seg_tool, tree) {
                    (SegTool::LiveWire, Some(tree)) => {
                        let px = [
                            h[0].round().clamp(0.0, (dims[0] - 1) as f64) as usize,
                            h[1].round().clamp(0.0, (dims[1] - 1) as f64) as usize,
                        ];
                        screen.extend(
                            tree.path_to(px)
                                .into_iter()
                                .skip(1)
                                .map(|q| to_screen([q[0] as f64, q[1] as f64])),
                        );
                    }
                    _ => screen.push(to_screen(h)),
                }
            }
            if screen.len() >= 2 {
                painter.add(egui::Shape::line(screen, stroke));
            }
        }
        if matches!(self.seg_tool, SegTool::ContourBrush | SegTool::Nudge) {
            if let Some(h) = hover {
                // The radius in screen pixels along a row (a round brush in
                // millimetres is an ellipse in pixels of unequal sides).
                let rx = (self.brush_radius_mm as f64 / img.spacing[0].max(1e-6)) as f32
                    * on.width()
                    / dims[0].max(1) as f32;
                let ry = (self.brush_radius_mm as f64 / img.spacing[1].max(1e-6)) as f32
                    * on.height()
                    / dims[1].max(1) as f32;
                let c = to_screen(h);
                let ring: Vec<Pos2> = (0..48)
                    .map(|t| {
                        let a = t as f32 / 48.0 * std::f32::consts::TAU;
                        c + Vec2::new(rx * a.cos(), ry * a.sin())
                    })
                    .collect();
                painter.add(egui::Shape::closed_line(
                    ring,
                    egui::Stroke::new(1.0, color),
                ));
            }
        }
    }
}
