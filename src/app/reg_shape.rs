//! *Align by structures*, inside the Image registration section: the two
//! picked images registered on the surfaces of structures contoured on
//! both, the voxel values left out of it ([`crate::registration::shape`]).
//!
//! What this file adds to the section is deciding *which structures*: the
//! structures drawn on the fixed image and the ones drawn on the moving
//! image (a structure set or a segmentation series that references the
//! series, or one that names no series of the workspace), paired by name,
//! case-insensitively, plus any pair put together by hand where the two
//! sides name an organ differently. Everything else is the section's: the
//! two images, *Start from*, the effort of a B-spline refinement, and what
//! happens to a result (it lands as the active registration through
//! [`ViewerApp::install_registration`], with the fusion, the vector field,
//! the analysis and propagation working on it as on any other).

use std::collections::BTreeMap;

use super::reg_panel::sample_field;
use super::*;
use crate::registration::shape::{self, ShapeDof, ShapePair, ShapeReport, ShapeRequest};
use crate::registration::{Init, RegParams};
use crate::workflow::select::{self, Structure};

/// The deformable stage after the rigid fit.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum ShapeRefine {
    /// Keep the rigid result.
    #[default]
    None,
    Elastix,
    Plastimatch,
}

impl ShapeRefine {
    const ALL: [ShapeRefine; 3] = [
        ShapeRefine::None,
        ShapeRefine::Elastix,
        ShapeRefine::Plastimatch,
    ];

    fn label(self) -> &'static str {
        match self {
            ShapeRefine::None => "nothing (rigid)",
            ShapeRefine::Elastix => "B-spline (elastix)",
            ShapeRefine::Plastimatch => "B-spline (plastimatch)",
        }
    }

    fn method(self) -> Option<RegMethod> {
        match self {
            ShapeRefine::None => None,
            ShapeRefine::Elastix => Some(RegMethod::ElastixBSpline),
            ShapeRefine::Plastimatch => Some(RegMethod::PlastimatchBSpline),
        }
    }
}

/// What the sub-section remembers between frames.
pub(super) struct ShapeRegState {
    /// Ticked and weight per pair, keyed by the two names in lower case:
    /// the rows themselves are rebuilt every frame from what the two images
    /// carry, so a structure that disappears takes its row with it and one
    /// that comes back finds its tick again.
    choices: BTreeMap<(String, String), (bool, f64)>,
    /// Pairs put together by hand: (fixed name, moving name).
    extra: Vec<(String, String)>,
    add_fixed: String,
    add_moving: String,
    dof: ShapeDof,
    refine: ShapeRefine,
    /// B-spline control point spacing of the refinement, mm.
    grid_mm: f64,
    /// Dilation of each structure for its refinement and the analysis, mm.
    margin_mm: f64,
    robust: bool,
    robust_mm: f64,
    symmetric: bool,
}

impl Default for ShapeRegState {
    fn default() -> Self {
        ShapeRegState {
            choices: BTreeMap::new(),
            extra: Vec::new(),
            add_fixed: String::new(),
            add_moving: String::new(),
            dof: ShapeDof::Rigid,
            refine: ShapeRefine::None,
            grid_mm: 16.0,
            margin_mm: 10.0,
            robust: false,
            robust_mm: 3.0,
            symmetric: true,
        }
    }
}

/// One row of the pair table.
struct ShapeRow {
    fixed: String,
    moving: String,
    color: [u8; 3],
    /// Put together by hand (and removable).
    by_hand: bool,
}

impl ShapeRow {
    fn key(&self) -> (String, String) {
        (self.fixed.to_lowercase(), self.moving.to_lowercase())
    }

    fn label(&self) -> String {
        if self.fixed.eq_ignore_ascii_case(&self.moving) {
            self.fixed.clone()
        } else {
            format!("{} / {}", self.fixed, self.moving)
        }
    }
}

/// What the sub-section asks for, done after its borrows end.
pub(super) enum ShapeAction {
    /// Run; `true` refines the active registration instead of starting over.
    Run(bool),
    Add,
    Drop(usize),
}

/// Whether a structure set (or segmentation series) that references
/// `referenced` belongs to the image `uid` of `study`: it names that series,
/// or it names none the workspace has (an older set, a free segmentation),
/// which cannot be told apart and is offered for every image.
fn bound_to(study: &LoadedStudy, referenced: &str, uid: &str) -> bool {
    referenced == uid || referenced.is_empty() || !study.series.iter().any(|s| s.uid == referenced)
}

impl ViewerApp {
    /// The structures drawn on one image: name and colour, one per name
    /// (case-insensitive), in the data tree's order.
    fn shape_names(&self, pick: RegPick) -> Vec<(String, [u8; 3])> {
        let Some(study) = self.slots[pick.slot].study.as_ref() else {
            return Vec::new();
        };
        let Some(uid) = study.series.get(pick.series).map(|s| s.uid.clone()) else {
            return Vec::new();
        };
        let mut out: Vec<(String, [u8; 3])> = Vec::new();
        for e in select::list(study) {
            let referenced = match e.kind {
                select::Kind::Roi => &study.structure_sets[e.set].referenced_series_uid,
                select::Kind::Segment => &study.seg_series[e.set].referenced_series_uid,
            };
            if !bound_to(study, referenced, &uid) {
                continue;
            }
            if !out.iter().any(|(n, _)| n.eq_ignore_ascii_case(&e.name)) {
                out.push((e.name, e.color));
            }
        }
        out
    }

    /// One structure of one image, frozen for the worker: the last one of
    /// that name drawn on the image (an exact match before a
    /// case-insensitive one), as [`select::find`] picks.
    fn shape_structure(&self, pick: RegPick, name: &str) -> Option<Structure> {
        let study = self.slots[pick.slot].study.as_ref()?;
        let uid = study.series.get(pick.series)?.uid.clone();
        let on: Vec<select::Entry> = select::list(study)
            .into_iter()
            .filter(|e| {
                let referenced = match e.kind {
                    select::Kind::Roi => &study.structure_sets[e.set].referenced_series_uid,
                    select::Kind::Segment => &study.seg_series[e.set].referenced_series_uid,
                };
                bound_to(study, referenced, &uid)
            })
            .collect();
        let lower = name.to_lowercase();
        let e = on
            .iter()
            .rfind(|e| e.name == name)
            .or_else(|| on.iter().rfind(|e| e.name.to_lowercase() == lower))?;
        select::structure(study, e)
    }

    /// The pair table: every name both images carry, then the pairs made by
    /// hand whose two structures are still there.
    fn shape_rows(&self) -> Vec<ShapeRow> {
        let fixed = self.shape_names(self.reg_fixed);
        let moving = self.shape_names(self.reg_moving);
        let mut rows: Vec<ShapeRow> = fixed
            .iter()
            .filter_map(|(f, color)| {
                moving
                    .iter()
                    .find(|(m, _)| m.eq_ignore_ascii_case(f))
                    .map(|(m, _)| ShapeRow {
                        fixed: f.clone(),
                        moving: m.clone(),
                        color: *color,
                        by_hand: false,
                    })
            })
            .collect();
        for (f, m) in &self.reg_shape.extra {
            let Some((_, color)) = fixed.iter().find(|(n, _)| n == f) else {
                continue;
            };
            if !moving.iter().any(|(n, _)| n == m) {
                continue;
            }
            let row = ShapeRow {
                fixed: f.clone(),
                moving: m.clone(),
                color: *color,
                by_hand: true,
            };
            if !rows.iter().any(|r| r.key() == row.key()) {
                rows.push(row);
            }
        }
        rows
    }

    /// The name behind a *Start from* structure choice of the fixed image.
    fn reg_roi_name(&self, slot: usize, roi: RegRoi) -> Option<String> {
        match roi {
            RegRoi::Whole => None,
            RegRoi::Structure(i) => self.slots[slot]
                .active_structures()
                .and_then(|ss| ss.rois.get(i))
                .map(|r| r.name.clone()),
            RegRoi::Segmentation(i) => self.slots[slot].segs().get(i).map(|s| s.name.clone()),
        }
    }

    /// The sub-section's rows, inside the Image registration section.
    /// `both` is whether two images can be registered at all.
    pub(super) fn shape_section(&mut self, ui: &mut egui::Ui, both: bool) -> Option<ShapeAction> {
        let mut action = None;
        ui.weak(
            "Aligns the two images on the surfaces of structures contoured on both: each \
             surface is laid onto its partner's distance map, both ways. The voxel values \
             are not used, so it works across contrast, modality and scanner, wherever the \
             contours are trusted.",
        );
        let rows = self.shape_rows();
        let st = &mut self.reg_shape;
        if rows.is_empty() {
            ui.weak(
                "No structure is contoured on both images. Pair two structures by hand \
                 below, or contour one on each image first.",
            );
        } else {
            ui.horizontal(|ui| {
                if ui.small_button("All").clicked() {
                    for r in &rows {
                        st.choices.entry(r.key()).or_insert((false, 1.0)).0 = true;
                    }
                }
                if ui.small_button("None").clicked() {
                    for r in &rows {
                        st.choices.entry(r.key()).or_insert((false, 1.0)).0 = false;
                    }
                }
            });
            egui::ScrollArea::vertical()
                .id_salt("reg_shape_rows")
                .max_height(180.0)
                .show(ui, |ui| {
                    egui::Grid::new("reg_shape_grid")
                        .num_columns(4)
                        .spacing([6.0, 2.0])
                        .striped(true)
                        .show(ui, |ui| {
                            run_report::head(ui, "");
                            run_report::head(ui, "Structure");
                            run_report::head(ui, "Weight");
                            ui.label("");
                            ui.end_row();
                            for r in &rows {
                                let entry = st.choices.entry(r.key()).or_insert((false, 1.0));
                                ui.checkbox(&mut entry.0, "");
                                ui.horizontal(|ui| {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(10.0, 10.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(
                                        rect,
                                        2.0,
                                        Color32::from_rgb(r.color[0], r.color[1], r.color[2]),
                                    );
                                    ui.label(r.label());
                                });
                                ui.add_enabled(
                                    entry.0,
                                    egui::DragValue::new(&mut entry.1)
                                        .speed(0.05)
                                        .range(0.1..=10.0)
                                        .max_decimals(2),
                                )
                                .on_hover_text(
                                    "How much this structure counts against the others. \
                                     Each structure is already normalised by its own size, \
                                     so 1 everywhere means every structure counts the same.",
                                );
                                if r.by_hand {
                                    if ui
                                        .small_button("🗑")
                                        .on_hover_text("Remove this pair")
                                        .clicked()
                                    {
                                        let at = st
                                            .extra
                                            .iter()
                                            .position(|(f, m)| *f == r.fixed && *m == r.moving);
                                        if let Some(i) = at {
                                            action = Some(ShapeAction::Drop(i));
                                        }
                                    }
                                } else {
                                    ui.label("");
                                }
                                ui.end_row();
                            }
                        });
                });
        }

        // ---- a pair by hand ----
        let fixed_names = self.shape_names(self.reg_fixed);
        let moving_names = self.shape_names(self.reg_moving);
        let st = &mut self.reg_shape;
        egui::CollapsingHeader::new("Pair by hand")
            .id_salt("reg_shape_by_hand")
            .default_open(false)
            .show(ui, |ui| {
                ui.weak("For one organ named differently on the two images (Heart / heart_total).");
                form::form(ui, "reg_shape_hand", |f| {
                    f.row("Fixed", |ui| {
                        egui::ComboBox::from_id_salt("reg_shape_add_fixed")
                            .selected_text(st.add_fixed.clone())
                            .width(170.0)
                            .show_ui(ui, |ui| {
                                for (n, _) in &fixed_names {
                                    ui.selectable_value(&mut st.add_fixed, n.clone(), n.as_str());
                                }
                            });
                    });
                    f.row("Moving", |ui| {
                        egui::ComboBox::from_id_salt("reg_shape_add_moving")
                            .selected_text(st.add_moving.clone())
                            .width(170.0)
                            .show_ui(ui, |ui| {
                                for (n, _) in &moving_names {
                                    ui.selectable_value(&mut st.add_moving, n.clone(), n.as_str());
                                }
                            });
                    });
                    f.wide(|ui| {
                        let ok = fixed_names.iter().any(|(n, _)| *n == st.add_fixed)
                            && moving_names.iter().any(|(n, _)| *n == st.add_moving);
                        if ui
                            .add_enabled(ok, egui::Button::new("➕ Add pair"))
                            .clicked()
                        {
                            action = Some(ShapeAction::Add);
                        }
                    });
                });
            });

        // ---- how ----
        form::form(ui, "reg_shape_form", |f| {
            f.row_tip(
                "Transform",
                "Rigid recovers three rotations and three translations; translation only \
                 keeps the rotations at zero (a couch shift between two scans).",
                |ui| {
                    egui::ComboBox::from_id_salt("reg_shape_dof")
                        .selected_text(st.dof.label())
                        .width(170.0)
                        .show_ui(ui, |ui| {
                            for d in ShapeDof::ALL {
                                ui.selectable_value(&mut st.dof, d, d.label());
                            }
                        });
                },
            );
            f.row_tip(
                "Then refine",
                "A local B-spline refinement on each structure's distance maps, from the \
                 largest to the smallest: every structure laid onto its partner, the rest of \
                 the patient left on the rigid result. Its resolutions, iterations and \
                 samples are the ones under Parameters.",
                |ui| {
                    egui::ComboBox::from_id_salt("reg_shape_refine")
                        .selected_text(st.refine.label())
                        .width(170.0)
                        .show_ui(ui, |ui| {
                            for r in ShapeRefine::ALL {
                                ui.selectable_value(&mut st.refine, r, r.label());
                            }
                        });
                },
            );
            if st.refine != ShapeRefine::None {
                f.row_tip(
                    "Grid",
                    "B-spline control point spacing of the refinement",
                    |ui| {
                        ui.add(
                            egui::DragValue::new(&mut st.grid_mm)
                                .speed(1.0)
                                .range(4.0..=60.0)
                                .suffix(" mm"),
                        );
                    },
                );
            }
            f.row_tip(
                "Margin",
                "Each structure grown by this much bounds its refinement, and the analysis \
                 of the result looks inside the structures grown by it.",
                |ui| {
                    ui.add(
                        egui::DragValue::new(&mut st.margin_mm)
                            .speed(1.0)
                            .range(0.0..=60.0)
                            .suffix(" mm"),
                    );
                },
            );
            f.row_tip(
                "Robust",
                "Distances beyond this width count linearly, not squared (Huber), so a \
                 slice contoured differently on one image does not steer the whole fit.",
                |ui| {
                    ui.checkbox(&mut st.robust, "");
                    ui.add_enabled(
                        st.robust,
                        egui::DragValue::new(&mut st.robust_mm)
                            .speed(0.1)
                            .range(0.5..=20.0)
                            .suffix(" mm"),
                    );
                },
            );
            f.row_tip(
                "Both ways",
                "Also lay the moving surfaces onto the fixed structures. Keeps a structure \
                 contoured over a shorter length on one image from sliding along its \
                 partner.",
                |ui| {
                    ui.checkbox(&mut st.symmetric, "");
                },
            );
            f.wide(|ui| {
                ui.weak(
                    "Starts from the structures' centroids matched (Start from: Identity \
                     keeps the identity; a structure there matches its own centroids).",
                );
            });
        });

        // ---- run ----
        let ticked = rows
            .iter()
            .filter(|r| st.choices.get(&r.key()).is_some_and(|c| c.0))
            .count();
        let fixed_uid = self.pick_series(self.reg_fixed).map(|(s, _)| s.uid);
        let moving_uid = self.pick_series(self.reg_moving).map(|(s, _)| s.uid);
        let can_refine = self.reg_shape.refine != ShapeRefine::None
            && self.registration.as_ref().is_some_and(|r| {
                Some(&r.fixed_uid) == fixed_uid.as_ref()
                    && Some(&r.moving_uid) == moving_uid.as_ref()
                    && r.fixed_slot == self.reg_fixed.slot
                    && r.moving_slot == self.reg_moving.slot
            });
        ui.horizontal(|ui| {
            if enabled_tip_button(
                ui,
                both && ticked > 0,
                format!("▶ Align on {ticked}"),
                "Register the moving image onto the fixed one by the ticked structures",
            ) {
                action = Some(ShapeAction::Run(false));
            }
            if enabled_tip_button(
                ui,
                both && ticked > 0 && can_refine,
                "▶ Refine active",
                "Skip the rigid fit and refine the active registration of these two \
                 images on the ticked structures",
            ) {
                action = Some(ShapeAction::Run(true));
            }
        });
        if ticked == 0 && !rows.is_empty() {
            ui.weak("Tick the structures to align on.");
        }
        action
    }

    /// Do what the sub-section asked.
    pub(super) fn shape_action(&mut self, a: ShapeAction) {
        match a {
            ShapeAction::Run(refine) => self.start_shape_registration(refine),
            ShapeAction::Add => {
                let st = &mut self.reg_shape;
                let pair = (st.add_fixed.clone(), st.add_moving.clone());
                let key = (pair.0.to_lowercase(), pair.1.to_lowercase());
                if !st.extra.contains(&pair) {
                    st.extra.push(pair);
                }
                st.choices.insert(key, (true, 1.0));
            }
            ShapeAction::Drop(i) => {
                if i < self.reg_shape.extra.len() {
                    self.reg_shape.extra.remove(i);
                }
            }
        }
    }

    /// Start a registration by structures between the two picked images
    /// (or a refinement of the active one). The structures are frozen
    /// here; the images, when not on display, are loaded on the worker,
    /// and the structures rasterized there onto whichever lattice each
    /// side has.
    fn start_shape_registration(&mut self, refine_active: bool) {
        if self.reg_job.is_some() {
            return;
        }
        self.settle_reg_picks();
        let (fpick, mpick) = (self.reg_fixed, self.reg_moving);
        if fpick == mpick {
            self.error = Some("The fixed and the moving image are the same series.".into());
            return;
        }
        let (Some((fseries, fshown)), Some((mseries, mshown))) =
            (self.pick_series(fpick), self.pick_series(mpick))
        else {
            self.error = Some("Pick a fixed and a moving image.".into());
            return;
        };
        // The ticked rows, each with its two structures frozen.
        let mut chosen: Vec<(String, [u8; 3], f64, Structure, Structure)> = Vec::new();
        for r in self.shape_rows() {
            let Some(&(true, weight)) = self.reg_shape.choices.get(&r.key()) else {
                continue;
            };
            let (Some(fs), Some(ms)) = (
                self.shape_structure(fpick, &r.fixed),
                self.shape_structure(mpick, &r.moving),
            ) else {
                continue;
            };
            chosen.push((r.label(), r.color, weight, fs, ms));
        }
        if chosen.is_empty() {
            self.error = Some("Tick at least one structure contoured on both images.".into());
            return;
        }
        // Start from: the identity, the structures' centroids, or the
        // centroids of one structure named in the section's own row.
        let mut init = match self.reg_init {
            RegInit::Identity => Init::Identity,
            _ => Init::Auto,
        };
        let mut init_by: Option<(String, Structure, Structure)> = None;
        if let RegInit::Structure(roi) = self.reg_init {
            if let Some(name) = self.reg_roi_name(fpick.slot, roi) {
                match (
                    self.shape_structure(fpick, &name),
                    self.shape_structure(mpick, &name),
                ) {
                    (Some(a), Some(b)) => init_by = Some((name, a, b)),
                    _ => {
                        self.error = Some(format!(
                            "Start from: '{name}' is not contoured on both images"
                        ));
                        return;
                    }
                }
            }
            init = Init::Auto;
        }
        let st = &self.reg_shape;
        let start = if refine_active {
            match &self.registration {
                Some(r) => Some(r.result.transform.clone()),
                None => {
                    self.error = Some("There is no active registration to refine.".into());
                    return;
                }
            }
        } else {
            None
        };
        let refine = st.refine.method().map(|method| RegParams {
            method,
            grid_spacing_mm: st.grid_mm,
            region: None,
            start: None,
            landmarks: Vec::new(),
            ..self.current_reg_params(None, false)
        });
        let request = ShapeRequest {
            pairs: Vec::new(),
            dof: st.dof,
            symmetric: st.symmetric,
            robust_mm: st.robust.then_some(st.robust_mm),
            init,
            start,
            refine,
            margin_mm: st.margin_mm,
            ..ShapeRequest::default()
        };
        let fixed_ready = fshown.then(|| self.pick_displayed_volume(fpick)).flatten();
        let moving_ready = mshown.then(|| self.pick_displayed_volume(mpick)).flatten();
        let fixed_slot = fpick.slot;
        let step = self.field_step_mm;
        let warp_only = self.field_warp_only;
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.reg_job = Some(Job::spawn(progress, move |p| {
            let load = |ready: Option<Arc<Volume>>, se: &loader::SeriesInfo, what: &str| match ready
            {
                Some(v) => Ok(v),
                None => {
                    p.set(format!("Loading the {what} image"));
                    loader::load_series_volume(se, p).map(|(v, _, _)| Arc::new(v))
                }
            };
            let run = || -> anyhow::Result<RegOutcome> {
                let fixed = load(fixed_ready, &fseries, "fixed")?;
                let moving = load(moving_ready, &mseries, "moving")?;
                let (fgrid, mgrid) = (fixed.grid(), moving.grid());
                p.set("Rasterizing the structures");
                let mut req = request;
                for (name, color, weight, fs, ms) in &chosen {
                    req.pairs.push(ShapePair {
                        name: name.clone(),
                        color: *color,
                        weight: *weight,
                        fixed: fs
                            .mask_on(&fgrid)
                            .map_err(|_| anyhow::anyhow!("'{name}' is empty on the fixed image"))?,
                        moving: ms.mask_on(&mgrid).map_err(|_| {
                            anyhow::anyhow!("'{name}' is empty on the moving image")
                        })?,
                    });
                }
                if let Some((name, a, b)) = &init_by {
                    let fc = a
                        .mask_on(&fgrid)
                        .ok()
                        .and_then(|m| crate::motion::centroid_mm(&m, &fgrid));
                    let mc = b
                        .mask_on(&mgrid)
                        .ok()
                        .and_then(|m| crate::motion::centroid_mm(&m, &mgrid));
                    match (fc, mc) {
                        (Some(fixed), Some(moving)) => req.init = Init::Points { fixed, moving },
                        _ => anyhow::bail!("Start from: '{name}' is empty on one of the images"),
                    }
                }
                let out = shape::register(&fixed, &moving, &req, p)?;
                p.set("Sampling the vector field");
                let field = sample_field(
                    &fixed,
                    &out.result.transform,
                    out.region.as_deref(),
                    step,
                    warp_only,
                );
                Ok(RegOutcome {
                    result: out.result,
                    field,
                    region: out.region,
                    fixed: RegImage {
                        slot: fpick.slot,
                        uid: fseries.uid.clone(),
                        vol: fixed,
                    },
                    moving: RegImage {
                        slot: mpick.slot,
                        uid: mseries.uid.clone(),
                        vol: moving,
                    },
                    shape: Some(out.report),
                })
            };
            (fixed_slot, run())
        }));
    }
}

/// The per-structure table of a registration by structures, under the
/// result.
pub(super) fn shape_report_rows(ui: &mut egui::Ui, rep: &ShapeReport) {
    egui::CollapsingHeader::new("By structures")
        .id_salt("reg_shape_report")
        .default_open(true)
        .show(ui, |ui| {
            let mut how = vec![rep.dof.label().to_string()];
            if rep.symmetric {
                how.push("both ways".into());
            }
            if let Some(d) = rep.robust_mm {
                how.push(format!("robust {d:.1} mm"));
            }
            if let Some(m) = rep.refine {
                how.push(format!("then {}", m.short()));
            }
            if rep.rigid_iterations > 0 {
                how.push(format!("{} rigid iterations", rep.rigid_iterations));
            }
            ui.weak(how.join(", "));
            egui::ScrollArea::horizontal()
                .id_salt("reg_shape_report_rows")
                .show(ui, |ui| {
                    egui::Grid::new("reg_shape_report_grid")
                        .striped(true)
                        .spacing([10.0, 2.0])
                        .show(ui, |ui| {
                            run_report::head(ui, "Structure");
                            run_report::head(ui, "Mean mm ▶");
                            run_report::head(ui, "Dice ▶");
                            run_report::head(ui, "Weight");
                            ui.end_row();
                            for l in &rep.lines {
                                ui.horizontal(|ui| {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(10.0, 10.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(
                                        rect,
                                        2.0,
                                        Color32::from_rgb(l.color[0], l.color[1], l.color[2]),
                                    );
                                    ui.label(&l.name);
                                });
                                let mut tip = format!(
                                    "Mean surface distance {:.2} ▶ {:.2} mm, RMS {:.2} ▶ {:.2} mm, \
                                     {} surface points",
                                    l.mean_before_mm,
                                    l.mean_after_mm,
                                    l.rms_before_mm,
                                    l.rms_after_mm,
                                    l.points
                                );
                                if let Some(r) = &l.refine_line {
                                    tip.push_str(&format!("\nRefinement: {r}"));
                                }
                                if let (Some(p95), Some(fold)) =
                                    (l.displacement_p95_mm, l.folded_fraction)
                                {
                                    tip.push_str(&format!(
                                        "\nDisplacement p95 {p95:.1} mm, folded {:.2}%",
                                        fold * 100.0
                                    ));
                                }
                                ui.monospace(format!(
                                    "{:.2} ▶ {:.2}",
                                    l.mean_before_mm, l.mean_after_mm
                                ))
                                .on_hover_text(tip);
                                let fmt = |d: Option<f64>| {
                                    d.map(|d| format!("{d:.2}")).unwrap_or_else(|| "-".into())
                                };
                                ui.monospace(format!(
                                    "{} ▶ {}",
                                    fmt(l.dice_before),
                                    fmt(l.dice_after)
                                ));
                                ui.monospace(format!("{:.2}", l.weight));
                                ui.end_row();
                            }
                        });
                });
        });
}
