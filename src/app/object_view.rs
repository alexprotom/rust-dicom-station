//! A registration or a treatment record in the row's additional pane: what
//! the data tree names in one line, laid out in full.
//!
//! The data tree holds data, one row per object; what an object says is
//! read here, as a report is ([`super::report_view`]). A spatial
//! registration is its facts (what it is, when, what it references and
//! whether those are loaded), then each matrix with the frames it maps and
//! the buttons that apply it between two workspaces, and a deformation
//! grid's; a treatment record is its facts and the table of its beams. The
//! pane keeps the bar every pane has: maximize, detach, close.

use super::*;
use crate::extras::{SpatialReg, TreatRecord};
use crate::sessions;

/// What the additional pane shows of the data tree's own objects: an index
/// into `study.registrations` or `study.treat_records`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PaneObject {
    Registration(usize),
    Record(usize),
}

impl ViewerApp {
    /// Show `obj` of `slot` in the additional pane, putting the pane into
    /// the row when it is not there yet; on screen, as a click in the tree
    /// means.
    pub(super) fn show_object(&mut self, slot: usize, obj: PaneObject) {
        let k = super::planar::ADDITIONAL_PANE;
        let v = &mut self.slots[slot].planar[k];
        v.layers.clear();
        v.blend = 0.0;
        v.report = None;
        v.object = Some(obj);
        self.add_to_row(slot, PaneKind::Additional);
        self.additional_on_screen(slot);
    }

    /// The second click on the row of what the additional pane shows (a
    /// report, a registration, a record): it leaves the pane, and the pane
    /// the row unless it holds images.
    pub(super) fn hide_additional(&mut self, slot: usize) {
        let k = super::planar::ADDITIONAL_PANE;
        let v = &mut self.slots[slot].planar[k];
        v.report = None;
        v.object = None;
        if v.layers.is_empty() && self.row_layout(slot).contains(&PaneKind::Additional) {
            self.remove_from_row(slot, PaneKind::Additional);
        }
    }

    /// Is `obj` of `slot` the one on screen in the additional pane?
    pub(super) fn object_shown(&self, slot: usize, obj: PaneObject) -> bool {
        self.pane_object(slot, super::planar::ADDITIONAL_PANE) == Some(obj)
    }

    /// Registration or record `idx` of `slot` is gone (`registration` says
    /// which list): the pane that showed it shows nothing of its own again,
    /// and the ones after it are renumbered.
    pub(super) fn object_removed(&mut self, slot: usize, registration: bool, idx: usize) {
        let k = super::planar::ADDITIONAL_PANE;
        let v = &mut self.slots[slot].planar[k];
        let (i, rebuild): (usize, fn(usize) -> PaneObject) = match v.object {
            Some(PaneObject::Registration(i)) if registration => (i, PaneObject::Registration),
            Some(PaneObject::Record(i)) if !registration => (i, PaneObject::Record),
            _ => return,
        };
        if i == idx {
            v.object = None;
            if v.layers.is_empty()
                && v.report.is_none()
                && self.row_layout(slot).contains(&PaneKind::Additional)
            {
                self.remove_from_row(slot, PaneKind::Additional);
            }
        } else if i > idx {
            v.object = Some(rebuild(i - 1));
        }
    }

    /// The additional pane showing `obj` of `slot`, filling `rect`.
    pub(super) fn object_pane(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        obj: PaneObject,
        rect: Rect,
    ) {
        let kind = PaneKind::Additional;
        let (fill, strong, weak) = {
            let v = ui.visuals();
            (v.panel_fill, v.strong_text_color(), v.weak_text_color())
        };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, fill);
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let (title, subtitle) = match obj {
            PaneObject::Registration(ri) => {
                let Some(reg) = study.registrations.get(ri) else {
                    return;
                };
                (
                    reg.label.clone(),
                    with_time(
                        format!("spatial registration · {}", reg_kind_label(study, reg)),
                        &reg.datetime,
                    ),
                )
            }
            PaneObject::Record(ri) => {
                let Some(rec) = study.treat_records.get(ri) else {
                    return;
                };
                (
                    rec.label.clone(),
                    with_time("treatment record".to_string(), &rec.datetime),
                )
            }
        };
        let title = if self.comparing() {
            format!("{title} · {}", SLOT_NAMES[slot])
        } else {
            title
        };

        // ---- the bar: as every pane has ----
        let is_max = self.pane_is_max(ui, slot, kind);
        let bsize = egui::vec2(24.0, 20.0);
        let by = rect.top() + 4.0;
        let cell = |right: f32| Rect::from_min_size(Pos2::new(right - bsize.x, by), bsize);
        let max_rect = cell(rect.right() - 4.0);
        let close_rect = cell(max_rect.left() - 4.0);
        let detach_rect = cell(close_rect.left() - 4.0);
        let max_resp = ui
            .put(
                max_rect,
                egui::Button::new(if is_max { "⊞" } else { "⛶" }).small(),
            )
            .on_hover_text(if is_max {
                "Restore the multi-view layout"
            } else {
                "Maximize this pane to the whole window"
            });
        let close_resp = ui
            .put(close_rect, egui::Button::new("✖").small())
            .on_hover_text("Take this pane out of the row");
        let clicked_detach = self.detach_button(ui, detach_rect, slot, kind);

        // ---- the title ----
        let title_h = 40.0;
        if self.show_labels {
            let room = rect.width() - 100.0;
            let fitted = |text: String, size: f32, color: Color32| {
                let mut job = egui::text::LayoutJob::single_section(
                    text,
                    egui::TextFormat::simple(FontId::proportional(size), color),
                );
                job.wrap = egui::text::TextWrapping::truncate_at_width(room.max(40.0));
                painter.layout_job(job)
            };
            painter.galley(
                rect.left_top() + Vec2::new(6.0, 4.0),
                fitted(title, 14.0, strong),
                strong,
            );
            painter.galley(
                rect.left_top() + Vec2::new(6.0, 22.0),
                fitted(subtitle, 12.0, weak),
                weak,
            );
        }

        // ---- the content ----
        let body = Rect::from_min_max(
            Pos2::new(rect.left() + 6.0, rect.top() + title_h),
            Pos2::new(rect.right() - 6.0, rect.bottom() - 6.0),
        );
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(body)
                .layout(egui::Layout::top_down(egui::Align::Min)),
        );
        child.set_clip_rect(body.intersect(ui.clip_rect()));
        egui::ScrollArea::both()
            .id_salt(("object_pane", slot))
            .auto_shrink([false, false])
            .show(&mut child, |ui| {
                ui.add_space(2.0);
                match obj {
                    PaneObject::Registration(ri) => self.reg_object_body(ui, slot, ri),
                    PaneObject::Record(ri) => self.record_object_body(ui, slot, ri),
                }
            });

        // ---- apply ----
        let resp = ui.interact(
            Rect::from_min_max(rect.min, Pos2::new(rect.right(), rect.top() + title_h)),
            egui::Id::new(("object_pane_title", slot)),
            Sense::hover(),
        );
        if resp.hovered() {
            self.hovered_slot = slot;
        }
        if max_resp.clicked() {
            self.toggle_pane_max(ui, slot, kind);
        }
        if clicked_detach {
            self.toggle_detached(slot, kind);
        }
        if close_resp.clicked() {
            self.remove_from_row(slot, kind);
        }
    }

    /// Registration `ri` of `slot` in full: its facts, then its matrices
    /// with the frames they map and the apply buttons, then a deformation
    /// grid's.
    fn reg_object_body(&mut self, ui: &mut egui::Ui, slot: usize, ri: usize) {
        // The workspaces this one can be paired with: a transform read
        // from a file maps one frame of reference onto another, and both
        // ends have to be on screen with an image in them.
        let partners: Vec<usize> = self
            .volume_slot_list()
            .into_iter()
            .filter(|s| *s != slot)
            .collect();
        // Frame-of-reference UID of every open workspace's volume, so the
        // hint can name the one a registration actually refers to.
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
        // (transform, fixed workspace, moving workspace) - with more than
        // two open, which pair a file's transform is meant for is a choice.
        let mut apply: Option<(registration::RigidTransform, usize, usize)> = None;
        let mut apply_grid: Option<(usize, usize)> = None;
        let mut invert = self.reg_apply_invert;
        {
            let Some(study) = self.slots[slot].study.as_ref() else {
                return;
            };
            let Some(reg) = study.registrations.get(ri) else {
                return;
            };

            // ---- the facts ----
            let mut facts: Vec<(String, String)> = Vec::new();
            facts.push(("Kind".into(), reg_kind_label(study, reg)));
            if !reg.datetime.is_empty() {
                facts.push(("Made".into(), stamp_label(&reg.datetime)));
            }
            if !reg.comment.is_empty() && reg.comment != reg.label {
                facts.push(("Comment".into(), reg.comment.clone()));
            }
            if let Some(n) = reg.series_number {
                facts.push(("Series number".into(), n.to_string()));
            }
            if reg.deformable {
                facts.push(("Type".into(), "deformable: matrices only".into()));
            }
            if sessions::reg_kind(reg, study) == sessions::RegKind::Treatment {
                if let Some(c) = sessions::correction(reg) {
                    facts.push(("Correction".into(), sessions::correction_text(&c)));
                }
            }
            for (k, u) in reg.references.iter().enumerate() {
                facts.push((
                    if k == 0 { "References" } else { "" }.into(),
                    reference_label(study, u),
                ));
            }
            if !reg.frame_of_reference_uid.is_empty() {
                facts.push((
                    "Frame of reference".into(),
                    format!(
                        "…{}{}",
                        tail(&reg.frame_of_reference_uid),
                        hint(&reg.frame_of_reference_uid)
                    ),
                ));
            }
            facts_grid(ui, ("reg_facts", slot, ri), &facts);

            // ---- the matrices ----
            for (ii, item) in reg.items.iter().enumerate() {
                if item.is_identity {
                    ui.weak(format!(
                        "Matrix {}: identity ({})",
                        ii + 1,
                        item.matrix_type
                    ));
                    continue;
                }
                let m = &item.matrix;
                ui.strong(format!("Matrix {}: {}", ii + 1, item.matrix_type));
                for r in 0..3 {
                    ui.monospace(format!(
                        "[{:7.3} {:7.3} {:7.3} {:8.2}]",
                        m[r * 4],
                        m[r * 4 + 1],
                        m[r * 4 + 2],
                        m[r * 4 + 3]
                    ));
                }
                ui.weak(format!(
                    "maps FoR …{}{} ▶ …{}{}",
                    tail(&item.for_uid),
                    hint(&item.for_uid),
                    tail(&reg.frame_of_reference_uid),
                    hint(&reg.frame_of_reference_uid)
                ));
                match extras::matrix_to_rigid(m, invert) {
                    Some(rigid) => {
                        let rp = rigid.params();
                        ui.weak(format!(
                            "t = ({:.1}, {:.1}, {:.1}) mm  r = ({:.2}, {:.2}, {:.2})°{}",
                            rp[3],
                            rp[4],
                            rp[5],
                            rp[0].to_degrees(),
                            rp[1].to_degrees(),
                            rp[2].to_degrees(),
                            if invert { "  (inverted)" } else { "" }
                        ));
                        ui.horizontal_wrapped(|ui| {
                            ui.checkbox(&mut invert, "Invert").on_hover_text(
                                "Invert the matrix before applying (flip the mapping direction)",
                            );
                            // This workspace against each of the others,
                            // both ways round.
                            for other in &partners {
                                for (fixed, moving) in [(slot, *other), (*other, slot)] {
                                    let label = format!(
                                        "Apply as {} ▶ {}",
                                        SLOT_NAMES[moving], SLOT_NAMES[fixed]
                                    );
                                    let tip = format!(
                                        "Use this matrix as the transform mapping {} (fixed) \
                                         coordinates into {} (moving)",
                                        SLOT_NAMES[fixed], SLOT_NAMES[moving]
                                    );
                                    if ui
                                        .add(egui::Button::new(label))
                                        .on_hover_text(tip)
                                        .clicked()
                                    {
                                        if let Some(r2) = extras::matrix_to_rigid(m, invert) {
                                            apply = Some((r2, fixed, moving));
                                        }
                                    }
                                }
                            }
                        });
                    }
                    None => {
                        ui.weak("The matrix is not a pure rigid transform - it cannot be applied");
                    }
                }
                ui.add_space(6.0);
            }
            // A Deformable Spatial Registration's displacement lattice
            // applies exactly as a matrix does.
            if let Some(grid) = &reg.grid {
                ui.strong("Deformation grid");
                ui.weak(grid.describe());
                let src = fors
                    .iter()
                    .find(|(_, f)| *f == reg.grid_source_for_uid)
                    .map(|(s, _)| *s);
                ui.horizontal_wrapped(|ui| {
                    for (fixed, moving) in partners.iter().flat_map(|o| [(slot, *o), (*o, slot)]) {
                        let label = format!(
                            "Apply grid as {} ▶ {}",
                            SLOT_NAMES[moving], SLOT_NAMES[fixed]
                        );
                        let tip = match src {
                            Some(s) if s == fixed => {
                                "The grid's own frame of reference matches this workspace - \
                                 this is the direction the file means"
                            }
                            Some(_) => {
                                "The grid's frame of reference matches the *other* workspace; \
                                 applying it this way round inverts what the file says"
                            }
                            None => {
                                "Neither loaded workspace matches the grid's frame of \
                                 reference - check that this is the right pair"
                            }
                        };
                        if enabled_tip_button(ui, true, label, tip) {
                            apply_grid = Some((fixed, moving));
                        }
                    }
                });
            }
            if partners.is_empty()
                && (reg.grid.is_some() || reg.items.iter().any(|i| !i.is_identity))
            {
                ui.add_space(4.0);
                ui.weak("Load a second workspace to apply this registration between the two");
            }
        }
        self.reg_apply_invert = invert;
        if let Some((rigid, fixed_slot, moving_slot)) = apply {
            self.apply_external_transform_between(
                Transform3::rigid_only(rigid),
                registration::RegMethod::ElastixRigid,
                fixed_slot,
                moving_slot,
            );
        }
        if let Some((fixed_slot, moving_slot)) = apply_grid {
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

    /// Record `ri` of `slot` in full: its facts, then its beams with the
    /// metersets specified and delivered and how each ended.
    fn record_object_body(&mut self, ui: &mut egui::Ui, slot: usize, ri: usize) {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(rec) = study.treat_records.get(ri) else {
            return;
        };
        let mut facts: Vec<(String, String)> = Vec::new();
        if !rec.datetime.is_empty() {
            facts.push(("Delivered".into(), stamp_label(&rec.datetime)));
        } else if !rec.date.is_empty() {
            facts.push(("Delivered".into(), rec.date.clone()));
        }
        let status = sessions::record_status(rec);
        if !status.is_empty() {
            facts.push(("Status".into(), status.to_string()));
        }
        if !rec.machine.is_empty() {
            facts.push(("Machine".into(), rec.machine.clone()));
        }
        if rec.ion {
            facts.push(("Radiation".into(), "ion".into()));
        }
        if let Some(f) = rec.fraction {
            facts.push(("Fraction (as recorded)".into(), f.to_string()));
        }
        if !rec.plan_uid.is_empty() {
            facts.push(("Plan".into(), reference_label(study, &rec.plan_uid)));
        }
        if let Some(n) = rec.series_number {
            facts.push(("Series number".into(), n.to_string()));
        }
        facts_grid(ui, ("record_facts", slot, ri), &facts);
        record_beams(ui, slot, ri, rec);
    }

    /// Shown means on screen: a workspace put aside comes back, and a
    /// maximized pane of another kind gives way to the row.
    pub(super) fn additional_on_screen(&mut self, slot: usize) {
        self.shown[slot] = true;
        if self
            .maximized
            .is_some_and(|(s, k)| s != slot || k != PaneKind::Additional)
        {
            self.maximized = None;
        }
    }
}

/// `text · 2024-09-17 10:54:03`, or `text` alone with no stamp.
fn with_time(text: String, stamp: &str) -> String {
    if stamp.is_empty() {
        text
    } else {
        format!("{text} · {}", stamp_label(stamp))
    }
}

/// `2024-09-17 10:54:03` of a `YYYYMMDDHHMMSS` stamp.
fn stamp_label(stamp: &str) -> String {
    format!(
        "{} {}",
        sessions::date_label(sessions::day(stamp)),
        sessions::clock(stamp)
    )
    .trim()
    .to_string()
}

/// What a registration is, in words: by what it references.
fn reg_kind_label(study: &LoadedStudy, reg: &SpatialReg) -> String {
    match sessions::reg_kind(reg, study) {
        sessions::RegKind::Treatment => "treatment position (references the plan)".into(),
        sessions::RegKind::CtToCt => "CT to CT (references two image series)".into(),
        sessions::RegKind::Imaging if reg.references.is_empty() => {
            "imaging position (references nothing)".into()
        }
        sessions::RegKind::Imaging => "imaging position".into(),
    }
}

/// A referenced UID as a person reads it: the plan or the series it names
/// when that is loaded, else that it is not.
fn reference_label(study: &LoadedStudy, uid: &str) -> String {
    if let Some(p) = study.plans.iter().find(|p| p.sop_instance_uid == uid) {
        let name = if p.label.is_empty() {
            &p.name
        } else {
            &p.label
        };
        return format!("plan {name}");
    }
    if let Some(s) = study.series.iter().find(|s| s.uid == uid) {
        let mut t = format!("{} series", s.modality);
        if let Some(n) = s.series_number {
            t.push_str(&format!(" #{n}"));
        }
        if !s.description.is_empty() {
            t.push_str(&format!(" {}", s.description));
        }
        if !s.series_datetime.is_empty() {
            t.push_str(&format!(" · {}", stamp_label(&s.series_datetime)));
        }
        return t;
    }
    format!("…{} (not loaded)", tail(uid))
}

/// Facts as a two-column grid; a fact with no name continues the one above.
fn facts_grid(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    facts: &[(String, String)],
) {
    if facts.is_empty() {
        return;
    }
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            for (k, v) in facts {
                ui.weak(k);
                ui.label(v);
                ui.end_row();
            }
        });
    ui.add_space(8.0);
}

/// The beams of a record: metersets specified and delivered, their
/// difference and how each ended (anything but `NORMAL` in the alert
/// colour).
fn record_beams(ui: &mut egui::Ui, slot: usize, ri: usize, rec: &TreatRecord) {
    if rec.beams.is_empty() {
        ui.weak("No beam in this record");
        return;
    }
    egui::Grid::new(("record_beams", slot, ri))
        .striped(true)
        .min_col_width(10.0)
        .spacing([14.0, 3.0])
        .show(ui, |ui| {
            for c in [
                "Beam",
                "No.",
                "MU spec",
                "MU del",
                "Δ%",
                "Status",
                "Verification",
            ] {
                ui.strong(c);
            }
            ui.end_row();
            for b in &rec.beams {
                ui.label(&b.name);
                ui.label(b.number.to_string());
                ui.label(
                    b.specified_meterset
                        .map(|m| format!("{m:.1}"))
                        .unwrap_or_else(|| "-".into()),
                );
                ui.label(
                    b.delivered_meterset
                        .map(|m| format!("{m:.1}"))
                        .unwrap_or_else(|| "-".into()),
                );
                ui.label(match (b.specified_meterset, b.delivered_meterset) {
                    (Some(s), Some(d)) if s > 1e-9 => format!("{:+.1}", 100.0 * (d - s) / s),
                    _ => "-".into(),
                });
                let status = if b.termination_status.is_empty() {
                    "-"
                } else {
                    &b.termination_status
                };
                if status == "NORMAL" || status == "-" {
                    ui.label(status);
                } else {
                    let c = alert_color(ui.visuals());
                    ui.label(egui::RichText::new(status).color(c));
                }
                ui.label(if b.verification_status.is_empty() {
                    "-"
                } else {
                    &b.verification_status
                });
                ui.end_row();
            }
        });
}
