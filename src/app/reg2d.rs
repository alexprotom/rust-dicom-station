//! *Modules ▶ Image registration ▶ 2D/2D registration*: one planar image
//! onto another - today's radiograph onto the reference image or the DRR
//! at the plan - by intensity, by pairs of points (placed, or markers
//! found), the points then the intensity, or deformably
//! ([`crate::registration::planar`]).
//!
//! The section chooses the fixed and the moving image (the image under a
//! pane's scrubber and the next one of its stack, to begin with), the
//! method and its parameters, and runs in the background. What comes back
//! is filed with the study (`study.planar_regs`, *2D registrations* in the
//! data tree) with the moving image resampled onto the fixed one as a
//! planar image of its own, stacked over the fixed image in a pane, where
//! the pane's compare modes show the agreement. The apply row carries the
//! moving image's planar structures onto the fixed one, reads two matches
//! of two views as one 3D correction (*⌖ As setup*, [`setup::from_planar`])
//! and saves the report.

use crate::registration::metric2d::Metric2d;
use crate::registration::planar::{
    self as p2, DeformMetric, Dof2d, Method2d, Optimizer2d, Planar2d, PlanarReg, Request2d,
    Result2d,
};
use crate::setup;

use super::*;

/// The section's glyph and name.
pub(super) const REG2D_TOOL: ToolInfo = ToolInfo {
    glyph: "⇄",
    name: "2D/2D registration",
};

/// The section's state.
pub(super) struct Reg2dState {
    pub(super) slot: usize,
    pub(super) fixed: Option<usize>,
    pub(super) moving: Option<usize>,
    pub(super) method: Method2d,
    pub(super) metric: Metric2d,
    pub(super) optimizer: Optimizer2d,
    pub(super) dof: Dof2d,
    pub(super) levels: usize,
    pub(super) pixels: usize,
    pub(super) range_mm: f64,
    pub(super) range_deg: f64,
    pub(super) refine: bool,
    /// Pairs of points: fixed pixel, moving pixel.
    pub(super) pairs: Vec<([f64; 2], [f64; 2])>,
    /// Find markers of this size, mm.
    pub(super) find_markers: bool,
    pub(super) marker_mm: f64,
    pub(super) grid_mm: f64,
    pub(super) grid_min_mm: f64,
    pub(super) lambda: f64,
    pub(super) deform_metric: DeformMetric,
    /// File the moved copy under the planar images.
    pub(super) file_copy: bool,
    /// *Place*: the next click on the fixed image starts a pair, the one
    /// after it on the moving image ends it.
    pub(super) placing: bool,
    pub(super) pending: Option<[f64; 2]>,
    /// The region of the fixed image the measure is taken over: polygons
    /// in its pixels; the whole image when empty. Drawn with `region_tool`
    /// on the fixed image, `drawing` the one under way.
    pub(super) region: Vec<Vec<[f64; 2]>>,
    pub(super) region_tool: RegionTool,
    pub(super) drawing: Vec<[f64; 2]>,
    /// Draw the current deformable registration's field over its fixed
    /// image: arrows of the warp on a lattice and the lattice deformed.
    pub(super) show_field: bool,
    /// The registration on display, within `study.planar_regs`.
    pub(super) current: Option<usize>,
    /// The registrations ticked for *As setup*.
    pub(super) for_setup: Vec<usize>,
    pub(super) status: Option<String>,
}

/// The region tools of the fixed image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum RegionTool {
    #[default]
    None,
    /// A drag from corner to corner.
    Rect,
    /// Click by click, closed with the right button or a double click.
    Polygon,
}

impl Reg2dState {
    fn new(slot: usize, prefs: &settings::Reg2dPrefs) -> Self {
        let deform_metric = DeformMetric::from_key(&prefs.deform_metric).unwrap_or_default();
        Reg2dState {
            slot,
            fixed: None,
            moving: None,
            method: Method2d::from_key(&prefs.method).unwrap_or_default(),
            metric: Metric2d::from_key(&prefs.metric).unwrap_or(Metric2d::GradientCorrelation),
            optimizer: Optimizer2d::Powell,
            dof: Dof2d::from_key(&prefs.dof).unwrap_or_default(),
            levels: 3,
            pixels: 256,
            range_mm: 30.0,
            range_deg: 10.0,
            refine: false,
            pairs: Vec::new(),
            find_markers: false,
            marker_mm: 2.0,
            grid_mm: (prefs.grid_mm as f64).clamp(4.0, 128.0),
            grid_min_mm: (prefs.grid_min_mm as f64).clamp(2.0, 64.0),
            lambda: prefs.lambda_milli as f64 / 1000.0,
            deform_metric,
            file_copy: true,
            placing: false,
            pending: None,
            region: Vec::new(),
            region_tool: RegionTool::None,
            drawing: Vec::new(),
            show_field: true,
            current: None,
            for_setup: Vec::new(),
            status: None,
        }
    }
}

impl Reg2dState {
    /// What is remembered of the section between runs.
    fn prefs(&self) -> settings::Reg2dPrefs {
        settings::Reg2dPrefs {
            method: self.method.key().into(),
            metric: self.metric.key().into(),
            dof: self.dof.key().into(),
            deform_metric: self.deform_metric.key().into(),
            grid_mm: self.grid_mm.round() as u16,
            grid_min_mm: self.grid_min_mm.round() as u16,
            lambda_milli: (self.lambda * 1000.0).round().max(0.0) as u32,
        }
    }
}

/// A finished run, with what it was asked so it can land after the
/// section changed.
pub(super) struct Reg2dOut {
    pub(super) result: Result2d,
    pub(super) fixed_uid: String,
    pub(super) moving_uid: String,
    pub(super) file_copy: bool,
}

/// What the pointer did on a pane with one of the section's tools in
/// hand, applied after the pane is drawn.
pub(super) enum PaneEdit2d {
    /// A click of *Place*.
    Place(usize, [f64; 2]),
    /// The rectangle: its first corner, then the corner under the pointer.
    RectFrom([f64; 2]),
    RectTo([f64; 2]),
    RectDone,
    /// A corner of the polygon, and its closing.
    Corner([f64; 2]),
    Close,
}

impl ViewerApp {
    /// Does any workspace hold two projection images?
    pub(super) fn reg2d_relevant(&self) -> bool {
        self.planar2d.reg_job.is_some()
            || self
                .open_slots()
                .into_iter()
                .any(|s| self.reg2d_images(s).len() >= 2)
    }

    /// The projection images of `slot`.
    fn reg2d_images(&self, slot: usize) -> Vec<usize> {
        self.slots[slot]
            .study
            .as_ref()
            .map(|st| {
                st.planar_images
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        p.kind == crate::extras::PlanarKind::Projection
                            && !p.data.is_empty()
                            && !p.sop_instance_uid.is_empty()
                    })
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Make the state, or point it at a workspace that has images.
    fn reg2d_follow(&mut self) {
        let slot = self
            .planar2d
            .reg
            .as_ref()
            .map(|s| s.slot)
            .filter(|&s| self.reg2d_images(s).len() >= 2)
            .or_else(|| {
                self.open_slots()
                    .into_iter()
                    .find(|&s| self.reg2d_images(s).len() >= 2)
            });
        let Some(slot) = slot else { return };
        if self.planar2d.reg.as_ref().is_none_or(|s| s.slot != slot) {
            let mut st = Reg2dState::new(slot, &self.planar2d.prefs);
            // The image under the scrubber of the first planar pane is the
            // fixed one, the next of its stack the moving one.
            let layers: Vec<usize> = self.slots[slot].planar[0]
                .layers
                .iter()
                .map(|l| l.image)
                .collect();
            let imgs = self.reg2d_images(slot);
            st.fixed = layers
                .first()
                .copied()
                .filter(|i| imgs.contains(i))
                .or(imgs.first().copied());
            st.moving = layers
                .get(1)
                .copied()
                .filter(|i| imgs.contains(i))
                .or_else(|| imgs.iter().copied().find(|i| Some(*i) != st.fixed));
            self.planar2d.reg = Some(st);
        }
    }

    /// Does one of the section's tools (*Place*, the region) take the left
    /// button over `image` of `slot`?
    pub(super) fn reg2d_pane_tool(&self, slot: usize, image: usize) -> bool {
        self.planar2d.reg.as_ref().is_some_and(|st| {
            st.slot == slot
                && ((st.placing
                    && match st.pending {
                        None => st.fixed == Some(image),
                        Some(_) => st.moving == Some(image),
                    })
                    || (st.region_tool != RegionTool::None && st.fixed == Some(image)))
        })
    }

    /// The pointer over `image` (`dims`, drawn in `on`) with a tool of the
    /// section in hand.
    pub(super) fn reg2d_pane_input(
        &self,
        image: usize,
        dims: [usize; 2],
        on: Rect,
        resp: &egui::Response,
    ) -> Vec<PaneEdit2d> {
        let Some(st) = &self.planar2d.reg else {
            return Vec::new();
        };
        let at = |mp: Pos2| super::planar_contours::screen_to_px(mp, dims, on);
        let pos = resp.interact_pointer_pos();
        let mut out = Vec::new();
        if st.placing {
            if let (true, Some(mp)) = (resp.clicked(), pos) {
                out.push(PaneEdit2d::Place(image, at(mp)));
            }
            return out;
        }
        match st.region_tool {
            RegionTool::Rect => {
                if resp.drag_started_by(egui::PointerButton::Primary) {
                    if let Some(mp) = pos {
                        out.push(PaneEdit2d::RectFrom(at(mp)));
                    }
                } else if resp.dragged_by(egui::PointerButton::Primary) {
                    if let Some(mp) = pos {
                        out.push(PaneEdit2d::RectTo(at(mp)));
                    }
                }
                if resp.drag_stopped_by(egui::PointerButton::Primary) {
                    out.push(PaneEdit2d::RectDone);
                }
            }
            RegionTool::Polygon => {
                if let (true, Some(mp)) = (resp.clicked(), pos) {
                    out.push(PaneEdit2d::Corner(at(mp)));
                }
                if resp.clicked_by(egui::PointerButton::Secondary) || resp.double_clicked() {
                    out.push(PaneEdit2d::Close);
                }
            }
            RegionTool::None => {}
        }
        out
    }

    /// Apply what the pointer did with the section's tools.
    pub(super) fn reg2d_pane_apply(&mut self, edits: Vec<PaneEdit2d>) {
        let Some(st) = self.planar2d.reg.as_mut() else {
            return;
        };
        for e in edits {
            match e {
                PaneEdit2d::Place(image, px) => match st.pending.take() {
                    None if st.fixed == Some(image) => st.pending = Some(px),
                    Some(f) if st.moving == Some(image) => st.pairs.push((f, px)),
                    p => st.pending = p,
                },
                PaneEdit2d::RectFrom(p) => st.drawing = vec![p, p],
                PaneEdit2d::RectTo(p) => {
                    if st.drawing.len() == 2 {
                        st.drawing[1] = p;
                    }
                }
                PaneEdit2d::RectDone => {
                    if let [a, b] = st.drawing[..] {
                        if (a[0] - b[0]).abs() > 1.0 && (a[1] - b[1]).abs() > 1.0 {
                            st.region.push(vec![a, [b[0], a[1]], b, [a[0], b[1]]]);
                        }
                    }
                    st.drawing.clear();
                }
                PaneEdit2d::Corner(p) => st.drawing.push(p),
                PaneEdit2d::Close => {
                    if st.drawing.len() >= 3 {
                        st.region.push(std::mem::take(&mut st.drawing));
                    }
                    st.drawing.clear();
                }
            }
        }
    }

    /// The pairs over a pane's image: numbered crosses on the fixed image
    /// and on the moving one.
    pub(super) fn reg2d_marks(&self, painter: &egui::Painter, slot: usize, image: usize, on: Rect) {
        let Some(st) = self.planar2d.reg.as_ref().filter(|st| st.slot == slot) else {
            return;
        };
        let Some(img) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|s| s.planar_images.get(image))
        else {
            return;
        };
        let dims = [img.cols, img.rows];
        let side = if st.fixed == Some(image) {
            0
        } else if st.moving == Some(image) {
            1
        } else {
            return;
        };
        let color = Color32::from_rgb(255, 160, 0);
        if st.show_field {
            self.reg2d_field(painter, slot, image, on);
        }
        // The region and the one being drawn, over the fixed image.
        if side == 0 {
            let to = |p: &[f64; 2]| super::planar_contours::px_to_screen(*p, dims, on);
            let stroke = egui::Stroke::new(1.5, Color32::from_rgb(0, 200, 120));
            for poly in &st.region {
                painter.add(egui::Shape::closed_line(
                    poly.iter().map(to).collect(),
                    stroke,
                ));
            }
            if st.drawing.len() >= 2 {
                let pts: Vec<Pos2> = if st.region_tool == RegionTool::Rect {
                    let (a, b) = (st.drawing[0], st.drawing[1]);
                    [a, [b[0], a[1]], b, [a[0], b[1]], a]
                        .iter()
                        .map(to)
                        .collect()
                } else {
                    st.drawing.iter().map(to).collect()
                };
                painter.add(egui::Shape::line(pts, stroke));
            }
        }
        let mut pts: Vec<(String, [f64; 2])> = st
            .pairs
            .iter()
            .enumerate()
            .map(|(k, p)| (format!("{}", k + 1), if side == 0 { p.0 } else { p.1 }))
            .collect();
        if let (0, Some(p)) = (side, st.pending) {
            pts.push((format!("{}", st.pairs.len() + 1), p));
        }
        for (name, p) in pts {
            let c = super::planar_contours::px_to_screen(p, dims, on);
            let r = 6.0;
            let s = egui::Stroke::new(1.5, color);
            painter.line_segment([c - Vec2::new(r, 0.0), c + Vec2::new(r, 0.0)], s);
            painter.line_segment([c - Vec2::new(0.0, r), c + Vec2::new(0.0, r)], s);
            painter.text(
                c + Vec2::new(r, -r),
                Align2::LEFT_BOTTOM,
                name,
                FontId::proportional(11.0),
                color,
            );
        }
    }

    /// The warp of the section's current registration over its fixed image
    /// `image` (drawn in `on`): an arrow per node of a lattice about 24
    /// screen pixels apart, from where a point is to where the warp sends
    /// it (the rigid part left out, at the fixed image's scale), and the
    /// lattice itself deformed.
    fn reg2d_field(&self, painter: &egui::Painter, slot: usize, image: usize, on: Rect) {
        let Some(st) = self.planar2d.reg.as_ref() else {
            return;
        };
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(reg) = st.current.and_then(|k| study.planar_regs.get(k)) else {
            return;
        };
        let Some(img) = study.planar_images.get(image) else {
            return;
        };
        if reg.fixed_uid != img.sop_instance_uid {
            return;
        }
        let Some(w) = &reg.result.transform.warp else {
            return;
        };
        let dims = [img.cols, img.rows];
        let screen_px = on.width() / dims[0].max(1) as f32;
        let step = ((24.0 / screen_px.max(1e-3)).round() as usize).max(4);
        let to = |p: [f64; 2]| super::planar_contours::px_to_screen(p, dims, on);
        let moved = |i: usize, j: usize| -> [f64; 2] {
            let p = [i as f64, j as f64];
            let d = w.displacement(p2::px_to_mm(dims, img.spacing, p));
            [
                p[0] + d[0] / img.spacing[0].max(1e-9),
                p[1] + d[1] / img.spacing[1].max(1e-9),
            ]
        };
        let grid = egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(80, 200, 255, 110));
        let arrow = egui::Stroke::new(1.5, Color32::from_rgb(255, 220, 60));
        let xs: Vec<usize> = (step / 2..dims[0]).step_by(step).collect();
        let ys: Vec<usize> = (step / 2..dims[1]).step_by(step).collect();
        for &j in &ys {
            let line: Vec<Pos2> = xs.iter().map(|&i| to(moved(i, j))).collect();
            painter.add(egui::Shape::line(line, grid));
        }
        for &i in &xs {
            let line: Vec<Pos2> = ys.iter().map(|&j| to(moved(i, j))).collect();
            painter.add(egui::Shape::line(line, grid));
        }
        for &j in &ys {
            for &i in &xs {
                let (a, b) = (to([i as f64, j as f64]), to(moved(i, j)));
                if (b - a).length() < 0.5 {
                    continue;
                }
                painter.arrow(a, b - a, arrow);
            }
        }
    }

    fn start_reg2d(&mut self) {
        if self.planar2d.reg_job.is_some() {
            return;
        }
        let Some(st) = &self.planar2d.reg else { return };
        let Some(study) = &self.slots[st.slot].study else {
            return;
        };
        let (Some(fi), Some(mi)) = (st.fixed, st.moving) else {
            self.error = Some("Choose the fixed and the moving image".into());
            return;
        };
        let (Some(f), Some(m)) = (study.planar_images.get(fi), study.planar_images.get(mi)) else {
            return;
        };
        let mut req = Request2d::new(Planar2d::of(f), Planar2d::of(m));
        req.method = st.method;
        req.metric = st.metric;
        req.optimizer = st.optimizer;
        req.dof = st.dof;
        req.levels = st.levels;
        req.max_pixels = st.pixels;
        req.range_mm = st.range_mm;
        req.range_deg = st.range_deg;
        req.refine = st.refine;
        req.pairs = st.pairs.clone();
        req.detect_markers = st.find_markers.then_some(st.marker_mm);
        if !st.region.is_empty() {
            req.mask = Some(setup::region_mask([f.cols, f.rows], &st.region));
        }
        req.grid_mm = st.grid_mm;
        req.grid_min_mm = st.grid_min_mm;
        req.lambda = st.lambda;
        req.deformable_metric = st.deform_metric;
        let (slot, file_copy) = (st.slot, st.file_copy);
        let (fixed_uid, moving_uid) = (f.sop_instance_uid.clone(), m.sop_instance_uid.clone());
        let progress = Arc::new(Progress::default());
        progress.set("registering");
        self.planar2d.reg_job = Some(Job::spawn(progress, move |p| {
            let r = p2::register(&req, p).map(|result| Reg2dOut {
                result,
                fixed_uid,
                moving_uid,
                file_copy,
            });
            (slot, r)
        }));
    }

    pub(super) fn poll_reg2d_job(&mut self, ctx: &egui::Context) {
        if let Some((slot, out)) = poll_tool_job(
            &mut self.planar2d.reg_job,
            ctx,
            REG2D_TOOL.name,
            &mut self.error,
        ) {
            self.on_reg2d_done(slot, out);
        }
    }

    /// A run landed: filed with the study, the moved copy stacked over the
    /// fixed image.
    fn on_reg2d_done(&mut self, slot: usize, out: Reg2dOut) {
        let Some(study) = self.slots[slot].study.as_mut() else {
            return;
        };
        let (Some(fi), Some(mi)) = (
            study.planar_by_uid(&out.fixed_uid),
            study.planar_by_uid(&out.moving_uid),
        ) else {
            self.error = Some(stale_result(&REG2D_TOOL));
            return;
        };
        let (fixed, moving) = (&study.planar_images[fi], &study.planar_images[mi]);
        let label = format!("{} onto {}", moving.label, fixed.label);
        let mut copy_uid = None;
        let mut copy_idx = None;
        if out.file_copy {
            let img = p2::resample(
                &Planar2d::of(moving),
                &out.result.transform,
                &Planar2d::of(fixed),
            );
            let (lo, hi) = img
                .data
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let uid = crate::dicom_export::new_uid();
            let copy = crate::extras::PlanarImage {
                label: label.clone(),
                modality: moving.modality.clone(),
                kind: crate::extras::PlanarKind::Projection,
                rows: fixed.rows,
                cols: fixed.cols,
                spacing: fixed.spacing,
                data: img.data,
                min_value: lo,
                max_value: hi,
                window: moving.window,
                info: vec![
                    (
                        "Source".into(),
                        format!("{} moved by a 2D/2D registration", moving.label),
                    ),
                    ("Transform".into(), out.result.line()),
                ],
                sop_instance_uid: uid.clone(),
                series_instance_uid: crate::dicom_export::new_uid(),
                frame_of_reference_uid: fixed.frame_of_reference_uid.clone(),
                study_uid: fixed.study_uid.clone(),
                iec: fixed.iec,
                camera: fixed.camera,
                megavoltage: fixed.megavoltage,
                frame_count: 1,
                ..Default::default()
            };
            study.planar_images.push(copy);
            copy_idx = Some(study.planar_images.len() - 1);
            copy_uid = Some(uid);
        }
        let line = out.result.line();
        study.planar_regs.push(PlanarReg {
            label,
            fixed_uid: out.fixed_uid,
            moving_uid: out.moving_uid,
            result: out.result,
            moved_copy_uid: copy_uid,
            when: crate::dicom_export::today().0,
        });
        let k = study.planar_regs.len() - 1;
        self.settings_gen += 1;
        if let Some(st) = &mut self.planar2d.reg {
            st.current = Some(k);
            st.status = Some(format!("✔ {line}"));
        }
        // The fixed image with the moved copy over it.
        self.show_planar(slot, fi, None);
        if let Some(c) = copy_idx {
            self.stack_planar(slot, c, None);
        }
        self.planar_on_screen(slot);
    }

    /// *⇥ Carry contours*: the planar structures on registration `k`'s
    /// moving image carried onto its fixed image, into a set named after it.
    pub(super) fn reg2d_carry(&mut self, slot: usize, k: usize, only: Option<(usize, usize)>) {
        let Some(study) = self.slots[slot].study.as_mut() else {
            return;
        };
        let Some(reg) = study.planar_regs.get(k).cloned() else {
            return;
        };
        let (Some(fi), Some(mi)) = (
            study.planar_by_uid(&reg.fixed_uid),
            study.planar_by_uid(&reg.moving_uid),
        ) else {
            self.error = Some("The registration's images are no longer in the workspace".into());
            return;
        };
        let (from, to) = (&study.planar_images[mi], &study.planar_images[fi]);
        let mut set = crate::planar_structs::PlanarStructureSet::new(
            &format!("Carried · {}", reg.label),
            &to.study_uid,
            "carried",
        );
        let mut refused = 0;
        for (si, s) in study.planar_structs.iter().enumerate() {
            for (j, st) in s.structures.iter().enumerate() {
                if only.is_some_and(|o| o != (si, j)) {
                    continue;
                }
                let Some(c) = st.contour(&reg.moving_uid) else {
                    continue;
                };
                match crate::planar_structs::carry(&c.region, &reg.result.transform, from, to) {
                    Some(region) if !region.is_empty() => {
                        let d = set.structure_entry(&st.name, Some(st.color));
                        let e = d.contour_entry(&reg.fixed_uid);
                        e.region = region;
                        e.source = crate::planar_structs::ContourSource::Carried {
                            from_image: reg.moving_uid.clone(),
                            registration: reg.label.clone(),
                        };
                    }
                    Some(_) => {}
                    None => refused += 1,
                }
            }
        }
        set.refresh_images();
        if set.structures.is_empty() {
            self.error = Some(if refused > 0 {
                "The warp folds where the contours are: nothing was carried".into()
            } else {
                "The moving image has no planar structure to carry".into()
            });
            return;
        }
        let n = set.structures.len();
        let idx = crate::planar_structs::file_set(study, set);
        self.slots[slot].active_planar = Some((idx, 0));
        self.settings_gen += 1;
        if let Some(st) = &mut self.planar2d.reg {
            st.status = Some(format!(
                "✔ {n} structure(s) carried{}",
                if refused > 0 {
                    format!(", {refused} refused inside a fold")
                } else {
                    String::new()
                }
            ));
        }
    }

    /// *⇥ Carry to...* from a planar structure's menu: through every 2D/2D
    /// registration whose moving image holds it.
    pub(super) fn open_carry(&mut self, slot: usize, set: usize, structure: usize) {
        let regs: Vec<usize> = self.slots[slot]
            .study
            .as_ref()
            .map(|st| {
                let Some(s) = st
                    .planar_structs
                    .get(set)
                    .and_then(|x| x.structures.get(structure))
                else {
                    return Vec::new();
                };
                st.planar_regs
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| s.contour(&r.moving_uid).is_some())
                    .map(|(k, _)| k)
                    .collect()
            })
            .unwrap_or_default();
        if regs.is_empty() {
            self.error = Some(
                "No 2D/2D registration of this workspace has this structure's image as its \
                 moving image"
                    .into(),
            );
            return;
        }
        for k in regs {
            self.reg2d_carry(slot, k, Some((set, structure)));
        }
    }

    /// *⌖ As setup*: the ticked registrations read as one 3D correction.
    fn reg2d_as_setup(&mut self) {
        let Some(st) = &self.planar2d.reg else { return };
        let slot = st.slot;
        let picks = st.for_setup.clone();
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let regs: Vec<&PlanarReg> = picks
            .iter()
            .filter_map(|&k| study.planar_regs.get(k))
            .collect();
        let opts = setup::SetupOptions {
            max_pixels: 256,
            ..Default::default()
        };
        match setup::from_planar(study, &regs, &opts, &crate::progress::Quiet) {
            Ok(s) => {
                let line = s.result.correction_line();
                let mut k = None;
                if let Some(study) = self.slots[slot].study.as_mut() {
                    study.setups.push(s);
                    k = Some(study.setups.len() - 1);
                }
                // *Refine in 3D*: the 2D/3D section starts from it.
                if let Some(r) = self.r2d.as_mut().filter(|r| r.slot == slot) {
                    r.start_setup = k;
                }
                self.settings_gen += 1;
                if let Some(st) = &mut self.planar2d.reg {
                    st.status = Some(format!("✔ filed under Setups: {line}"));
                }
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    /// Registration `k` of `slot` was removed from the study.
    pub(super) fn reg2d_removed(&mut self, slot: usize, k: usize) {
        if let Some(st) = self.planar2d.reg.as_mut().filter(|st| st.slot == slot) {
            st.current = match st.current {
                Some(c) if c == k => None,
                Some(c) if c > k => Some(c - 1),
                c => c,
            };
            st.for_setup.retain(|&c| c != k);
            for c in &mut st.for_setup {
                if *c > k {
                    *c -= 1;
                }
            }
        }
    }

    /// The section, inside the Image registration module.
    pub(super) fn reg2d_section(&mut self, ui: &mut egui::Ui) {
        self.reg2d_follow();
        let busy = self.planar2d.reg_job.is_some();
        let Some(mut st) = self.planar2d.reg.take() else {
            ui.weak("Load a workspace with two planar images");
            return;
        };
        let images = self.reg2d_images(st.slot);
        let Some(study) = self.slots[st.slot].study.as_ref() else {
            self.planar2d.reg = Some(st);
            return;
        };
        let label = |i: Option<usize>| {
            i.and_then(|i| study.planar_images.get(i))
                .map(|p| p.label.clone())
                .unwrap_or_else(|| "(pick)".into())
        };
        let mut run = false;
        let mut cancel = false;
        let mut carry = None;
        let mut as_setup = false;
        let mut save_report = None;
        form::form(ui, "reg2d_images", |f| {
            for (name, which) in [("Fixed", 0), ("Moving", 1)] {
                f.row(name, |ui| {
                    let cur = if which == 0 { st.fixed } else { st.moving };
                    egui::ComboBox::from_id_salt(("reg2d_img", which))
                        .selected_text(label(cur))
                        .width(200.0)
                        .show_ui(ui, |ui| {
                            for &i in &images {
                                let t = if which == 0 {
                                    &mut st.fixed
                                } else {
                                    &mut st.moving
                                };
                                ui.selectable_value(t, Some(i), label(Some(i)));
                            }
                        });
                    if which == 1 && ui.small_button("⇄").tip("Swap them").clicked() {
                        std::mem::swap(&mut st.fixed, &mut st.moving);
                        st.pairs = st.pairs.iter().map(|(a, b)| (*b, *a)).collect();
                    }
                });
            }
            let cams = (
                st.fixed
                    .and_then(|i| study.planar_images.get(i))
                    .and_then(|p| p.camera),
                st.moving
                    .and_then(|i| study.planar_images.get(i))
                    .and_then(|p| p.camera),
            );
            if let (Some(a), Some(b)) = cams {
                let differ = (a.source - b.source).length() > 1.0
                    || (a.center() - b.center()).length() > 1.0;
                f.wide(|ui| {
                    if differ {
                        ui.colored_label(ui.visuals().warn_fg_color, "cameras differ");
                    } else {
                        ui.weak("camera");
                    }
                });
            }
            f.row("Method", |ui| {
                egui::ComboBox::from_id_salt("reg2d_method")
                    .selected_text(st.method.label())
                    .show_ui(ui, |ui| {
                        for m in Method2d::ALL {
                            ui.selectable_value(&mut st.method, m, m.label());
                        }
                    });
            });
            if st.method != Method2d::Points {
                f.row("Measure", |ui| {
                    egui::ComboBox::from_id_salt("reg2d_metric")
                        .selected_text(st.metric.label())
                        .show_ui(ui, |ui| {
                            for m in Metric2d::ALL {
                                ui.selectable_value(&mut st.metric, m, m.label());
                            }
                        });
                });
                f.row("Search", |ui| {
                    for o in Optimizer2d::ALL {
                        ui.selectable_value(&mut st.optimizer, o, o.label());
                    }
                    ui.checkbox(&mut st.refine, "Refine");
                });
                f.row("Levels · pixels", |ui| {
                    ui.add(egui::DragValue::new(&mut st.levels).range(1..=5));
                    ui.add(
                        egui::DragValue::new(&mut st.pixels)
                            .range(64..=1024)
                            .speed(8.0),
                    );
                });
                f.row("Range", |ui| {
                    ui.add(
                        egui::DragValue::new(&mut st.range_mm)
                            .range(1.0..=100.0)
                            .suffix(" mm"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut st.range_deg)
                            .range(0.0..=45.0)
                            .suffix("°"),
                    );
                });
            }
            f.row("Freedom", |ui| {
                egui::ComboBox::from_id_salt("reg2d_dof")
                    .selected_text(st.dof.label())
                    .show_ui(ui, |ui| {
                        for d in Dof2d::ALL {
                            ui.selectable_value(&mut st.dof, d, d.label());
                        }
                    });
            });
            if st.method == Method2d::Deformable {
                f.row("Grid", |ui| {
                    ui.add(
                        egui::DragValue::new(&mut st.grid_mm)
                            .range(4.0..=128.0)
                            .suffix(" mm"),
                    );
                    ui.label("to");
                    ui.add(
                        egui::DragValue::new(&mut st.grid_min_mm)
                            .range(2.0..=64.0)
                            .suffix(" mm"),
                    );
                });
                f.row("Measure", |ui| {
                    for m in DeformMetric::ALL {
                        if ui
                            .selectable_value(&mut st.deform_metric, m, m.label())
                            .clicked()
                        {
                            st.lambda = m.default_lambda();
                        }
                    }
                });
                f.row("Smoothness", |ui| {
                    ui.add(
                        egui::DragValue::new(&mut st.lambda)
                            .range(0.0..=10.0)
                            .speed(0.005),
                    );
                });
            }
        });
        let mut region_took = false;
        if st.method != Method2d::Points {
            let mut took = false;
            ui.horizontal_wrapped(|ui| {
                ui.label("Region");
                for (t, glyph, tip) in [
                    (RegionTool::Rect, "⬚", "Rectangle: drag on the fixed image"),
                    (
                        RegionTool::Polygon,
                        "✏",
                        "Polygon: click the corners, double-click to close",
                    ),
                ] {
                    if ui
                        .selectable_label(st.region_tool == t, glyph)
                        .on_hover_text(tip)
                        .clicked()
                    {
                        st.region_tool = if st.region_tool == t {
                            RegionTool::None
                        } else {
                            t
                        };
                        st.drawing.clear();
                        st.placing = false;
                        took = st.region_tool != RegionTool::None;
                    }
                }
                if !st.region.is_empty() {
                    ui.weak(format!("{} polygon(s)", st.region.len()));
                    if ui.small_button("Clear").clicked() {
                        st.region.clear();
                    }
                } else {
                    ui.weak("the whole image");
                }
            });
            region_took = took;
        }
        if matches!(st.method, Method2d::Points | Method2d::PointsThenIntensity) {
            ui.horizontal_wrapped(|ui| {
                let on = st.placing;
                if ui
                    .selectable_label(on, "📌 Place")
                    .on_hover_text("Click fixed, then moving, for each pair")
                    .clicked()
                {
                    st.placing = !on;
                    st.pending = None;
                    st.region_tool = RegionTool::None;
                }
                ui.checkbox(&mut st.find_markers, "Find markers");
                if st.find_markers {
                    ui.add(
                        egui::DragValue::new(&mut st.marker_mm)
                            .range(0.5..=10.0)
                            .suffix(" mm"),
                    );
                }
                if !st.pairs.is_empty() && ui.small_button("Clear").clicked() {
                    st.pairs.clear();
                }
            });
            let residuals: Vec<f64> = st
                .current
                .and_then(|k| study.planar_regs.get(k))
                .map(|r| r.result.pairs.iter().map(|p| p.residual_mm).collect())
                .unwrap_or_default();
            let mut drop = None;
            for (k, (a, b)) in st.pairs.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.monospace(format!(
                        "{} ({:.1}, {:.1}) → ({:.1}, {:.1}){}",
                        k + 1,
                        a[0],
                        a[1],
                        b[0],
                        b[1],
                        residuals
                            .get(k)
                            .map(|r| format!("  {r:.2} mm"))
                            .unwrap_or_default()
                    ));
                    if ui
                        .small_button("✖")
                        .named("Remove this point pair")
                        .clicked()
                    {
                        drop = Some(k);
                    }
                });
            }
            if let Some(k) = drop {
                st.pairs.remove(k);
            }
        }
        ui.checkbox(&mut st.file_copy, "File the moved copy");
        ui.horizontal(|ui| {
            let ready = st.fixed.is_some() && st.moving.is_some() && st.fixed != st.moving;
            if busy {
                cancel = ui.button("Cancel").clicked();
            } else if ui
                .add_enabled(ready, egui::Button::new("▶ Register"))
                .clicked()
            {
                run = true;
            }
        });
        if let Some(job) = &self.planar2d.reg_job {
            ui.label(job.progress.get());
            ui.add(egui::ProgressBar::new(job.progress.frac()).show_percentage());
        }
        if let Some(s) = &st.status {
            ui.label(s.as_str());
        }
        // The registrations of the workspace: the current one in full, the
        // others to tick for a setup.
        if !study.planar_regs.is_empty() {
            ui.separator();
            for (k, r) in study.planar_regs.iter().enumerate() {
                ui.horizontal(|ui| {
                    let placed = study
                        .planar_by_uid(&r.fixed_uid)
                        .is_some_and(|i| study.planar_images[i].camera.is_some());
                    let mut on = st.for_setup.contains(&k);
                    if ui
                        .add_enabled(placed, egui::Checkbox::new(&mut on, ""))
                        .on_hover_text("Use in ⌖ As setup")
                        .on_disabled_hover_text("Its fixed image has no geometry")
                        .changed()
                    {
                        if on {
                            st.for_setup.push(k);
                        } else {
                            st.for_setup.retain(|&c| c != k);
                        }
                    }
                    if ui
                        .selectable_label(st.current == Some(k), r.label.as_str())
                        .clicked()
                    {
                        st.current = Some(k);
                    }
                });
            }
            if let Some(r) = st.current.and_then(|k| study.planar_regs.get(k)) {
                ui.label(egui::RichText::new(r.result.line()).monospace());
                ui.weak(format!(
                    "{} {:.4} → {:.4} · {} evaluations · {:.1} s",
                    r.result.metric.label(),
                    r.result.before,
                    r.result.after,
                    r.result.evaluations,
                    r.result.elapsed_secs
                ));
                for l in &r.result.levels {
                    ui.weak(format!(
                        "level {} · {} × {} px · {:.4} → {:.4}",
                        l.level + 1,
                        l.dims[0],
                        l.dims[1],
                        l.before,
                        l.after
                    ));
                }
                if r.result.transform.warp.is_some() {
                    ui.checkbox(&mut st.show_field, "Field")
                        .on_hover_text("Show the warp on the fixed image");
                }
                if let Some(w) = &r.result.warp_stats {
                    ui.weak(format!(
                        "warp mean {:.2} mm, largest {:.2} mm, Jacobian {:.2} .. {:.2}",
                        w.mean_mm, w.max_mm, w.jacobian_min, w.jacobian_max
                    ));
                }
                if let Some(w) = &r.result.warning {
                    ui.colored_label(ui.visuals().warn_fg_color, w.as_str());
                }
                let k = st.current.unwrap_or(0);
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("⇥ Carry contours")
                        .on_hover_text("Copy the moving contours onto the fixed image")
                        .clicked()
                    {
                        carry = Some(k);
                    }
                    if ui
                        .add_enabled(st.for_setup.len() >= 2, egui::Button::new("⌖ As setup"))
                        .on_hover_text("Make a 3D setup from the ticked matches")
                        .clicked()
                    {
                        as_setup = true;
                    }
                    if ui.button("💾 Report").clicked() {
                        save_report = Some(
                            serde_json::to_string_pretty(&r.result.json()).unwrap_or_default(),
                        );
                    }
                    if ui
                        .button("📋 Copy")
                        .on_hover_text("Copy the transform as text")
                        .clicked()
                    {
                        ui.ctx().copy_text(r.result.line());
                    }
                });
            }
        }
        let slot = st.slot;
        let prefs = st.prefs();
        let placing = st.placing;
        self.planar2d.reg = Some(st);
        if placing || region_took {
            // One section at a time holds the pane's left button.
            self.reg2d3d_drop_pane_tool();
        }
        if prefs != self.planar2d.prefs {
            self.planar2d.prefs = prefs;
            self.persist_settings();
        }
        if cancel {
            cancel_if(true, &self.planar2d.reg_job);
        }
        if run {
            self.start_reg2d();
        }
        if let Some(k) = carry {
            self.reg2d_carry(slot, k, None);
        }
        if as_setup {
            self.reg2d_as_setup();
        }
        if let Some(text) = save_report {
            const JSON: pick::Filter = pick::Filter {
                name: "JSON",
                exts: &["json"],
            };
            self.ask_save(
                "Save the 2D/2D registration report",
                "reg2d_report.json",
                None,
                Some(JSON),
                move |app, path| {
                    if let Err(e) = std::fs::write(&path, text) {
                        app.error = Some(format!("Writing {} failed: {e}", path.display()));
                    }
                },
            );
        }
    }
}
