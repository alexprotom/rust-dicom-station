//! The export window: one window for both datasets, in which what goes out,
//! what it is called and what UID it carries are all visible and all editable.
//!
//! The plan it edits lives in [`crate::export`]; this file is only its
//! surface. Two rules shape the layout. Every identifier is on screen, because
//! an export whose UIDs you cannot read is one you cannot file. And the tree
//! is the selection, so "export these two phases and the structure set" is
//! three tick boxes rather than three runs.

use crate::export::{
    DatasetNode, ExportPlan, Field, GroupNode, Layout, ObjKind, ObjNode, PatientNode, SeriesNode,
    StructFormat, StudyNode, UidMode,
};

use super::*;

/// Width of the value box of an identifier.
const UID_W: f32 = 330.0;
/// Width of the value box of a name.
const NAME_W: f32 = 210.0;

impl ViewerApp {
    /// Open the export window, building a fresh plan from what is loaded.
    pub(super) fn open_export_dialog(&mut self) {
        let a = self.slots[0].study.as_ref();
        let b = self.slots[1].study.as_ref();
        if a.is_none() && b.is_none() {
            return;
        }
        let params = dicom_export::ExportParams::for_study(a.or(b).expect("one is loaded"));
        self.export_plan = Some(ExportPlan::build([a, b], params));
        self.export_result = None;
        self.export_warnings.clear();
        self.export_open = true;
    }

    pub(super) fn export_window(&mut self, ctx: &egui::Context) {
        if !self.export_open {
            return;
        }
        // The plan is edited in place while the tree is drawn, so it is taken
        // out of `self` for the duration - the closure needs the rest of the
        // application too (the output folder, the running job).
        let Some(mut plan) = self.export_plan.take() else {
            self.export_open = false;
            return;
        };
        if plan.datasets.is_empty() {
            self.export_open = false;
            return;
        }

        let busy = self.export_job.is_some();
        let mut open = true;
        let mut browse = false;
        let mut do_export = false;
        let job_text = self.export_job.as_ref().map(|j| j.progress.get());
        let result = self.export_result.clone();
        let warnings = self.export_warnings.clone();
        let dir = &mut self.export_dir;

        detach::tool_window(
            ctx,
            "export",
            "💾 Export DICOM",
            &mut open,
            detach::WinOpts::size(940.0, 640.0),
            |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Write to").strong());
                    ui.add(
                        egui::TextEdit::singleline(dir)
                            .desired_width(560.0)
                            .hint_text("output folder (created if missing)"),
                    );
                    if ui.button("📂 Browse").clicked() {
                        browse = true;
                    }
                });
                ui.add_space(4.0);
                run_options(ui, &mut plan);
                ui.separator();

                let avail = (ui.available_height() - 108.0).max(160.0);
                egui::ScrollArea::vertical()
                    .id_salt("export_tree")
                    .max_height(avail)
                    .show(ui, |ui| {
                        // A converted object is a new instance, and its UID
                        // field has to show that before the run, not after.
                        let mut formats_changed = false;
                        for di in 0..plan.datasets.len() {
                            dataset_node(ui, &mut plan.datasets[di], &mut formats_changed);
                        }
                        if formats_changed {
                            plan.sync_format_uids();
                        }
                        ui.add_space(8.0);
                        common_tags(ui, &mut plan);
                    });

                ui.separator();
                let (n_series, n_obj) = plan.counts();
                ui.horizontal(|ui| {
                    if let Some(text) = &job_text {
                        ui.spinner();
                        ui.label(text);
                    } else if ui
                        .add_enabled(
                            !busy && (n_series + n_obj) > 0,
                            egui::Button::new("💾 Export"),
                        )
                        .on_hover_text("Write the selection into the output folder")
                        .clicked()
                    {
                        do_export = true;
                    }
                    ui.weak(format!(
                        "{n_series} image series, {n_obj} RT object(s) selected"
                    ));
                    if let Some(msg) = &result {
                        ui.label(msg);
                    }
                });
                if !warnings.is_empty() {
                    egui::ScrollArea::vertical()
                        .id_salt("export_warnings")
                        .max_height(70.0)
                        .show(ui, |ui| {
                            for w in &warnings {
                                ui.label(egui::RichText::new(format!("⚠ {w}")).weak());
                            }
                        });
                }
            },
        );

        self.export_open = open;
        self.export_plan = Some(plan);
        if !open {
            self.export_result = None;
        }
        if browse {
            if let Some(d) = Self::pick_folder("Select the export output folder") {
                self.export_dir = d.display().to_string();
            }
        }
        if do_export {
            self.start_export();
        }
    }
}

/// The three run-wide choices.
fn run_options(ui: &mut egui::Ui, plan: &mut ExportPlan) {
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Identifiers").strong());
        let mut mode = plan.uid_mode;
        ui.radio_value(&mut mode, UidMode::Keep, "keep the original UIDs")
            .on_hover_text(
                "The export is the same study: re-importing it where it came from updates \
                 that study, and references from objects outside the export still resolve.",
            );
        ui.radio_value(&mut mode, UidMode::New, "generate new UIDs")
            .on_hover_text(
                "The export is a new study that happens to look like this one - what you \
                 want when the edited copy has to live beside its source.",
            );
        if mode != plan.uid_mode {
            plan.set_uid_mode(mode);
        }
        ui.separator();
        ui.label(egui::RichText::new("Structures").strong());
        if ui
            .small_button("all RTSTRUCT")
            .on_hover_text("Write every set of structures as contours")
            .clicked()
        {
            plan.set_all_formats(StructFormat::RtStruct);
        }
        if ui
            .small_button("all SEG")
            .on_hover_text("Write every set of structures as binary masks")
            .clicked()
        {
            plan.set_all_formats(StructFormat::Seg);
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Folders").strong());
        ui.radio_value(&mut plan.layout, Layout::Tree, "patient / study / series");
        ui.radio_value(&mut plan.layout, Layout::StudyFolders, "one per study");
        ui.radio_value(&mut plan.layout, Layout::Flat, "flat");
        ui.separator();
        ui.checkbox(&mut plan.rerender_images, "Rewrite images from the voxels")
            .on_hover_text(
                "Off (recommended): the source files are copied with only the attributes \
                 below patched, so private tags, acquisition parameters and the exact pixel \
                 data survive.\nOn: the images are written afresh from the loaded volume - \
                 needed only when the voxels themselves were changed.",
            );
    });
}

/// A tri-state tick box: click sets everything below it.
fn tri_checkbox(ui: &mut egui::Ui, state: Option<bool>) -> Option<bool> {
    let mut on = state.unwrap_or(true);
    let resp = if state.is_none() {
        // egui has no indeterminate box; a partly selected node shows as
        // ticked and weakly marked, and one click clears it.
        ui.scope(|ui| {
            ui.visuals_mut().selection.bg_fill = ui.visuals().weak_text_color();
            ui.checkbox(&mut on, "")
        })
        .inner
        .on_hover_text("Part of this is selected")
    } else {
        ui.checkbox(&mut on, "")
    };
    if resp.changed() {
        // From partly selected, the first click selects the rest.
        return Some(if state.is_none() { true } else { on });
    }
    None
}

/// Draws one dataset. `formats` is raised when a structure format radio moved.
fn dataset_node(ui: &mut egui::Ui, d: &mut DatasetNode, formats: &mut bool) {
    let id = ui.make_persistent_id(("exp_ds", d.slot));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let title = format!("Dataset {}", d.label);
    let n = d.patients.len();
    state
        .show_header(ui, |ui| {
            if let Some(on) = tri_checkbox(ui, d.all_selected()) {
                d.set_all(on);
            }
            ui.label(egui::RichText::new(title).strong());
            ui.weak(format!("{n} patient(s)"));
        })
        .body(|ui| {
            for p in &mut d.patients {
                patient_node(ui, p, formats);
            }
        });
}

fn patient_node(ui: &mut egui::Ui, p: &mut PatientNode, formats: &mut bool) {
    let id = ui.make_persistent_id(("exp_pat", p.slot, &p.key));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let title = format!("👤 {}", label_of(&p.name, &p.id, &p.key));
    let n = p.studies.len();
    state
        .show_header(ui, |ui| {
            if let Some(on) = tri_checkbox(ui, p.all_selected()) {
                p.set_all(on);
            }
            ui.label(title);
            ui.weak(format!("{n} study(ies)"));
        })
        .body(|ui| {
            egui::Grid::new(id.with("f"))
                .num_columns(2)
                .spacing([8.0, 3.0])
                .show(ui, |ui| {
                    text_row(ui, "PatientName", &mut p.name, NAME_W);
                    text_row(ui, "PatientID", &mut p.id, NAME_W);
                });
            ui.add_space(3.0);
            for st in &mut p.studies {
                study_node(ui, st, formats);
            }
        });
}

fn study_node(ui: &mut egui::Ui, st: &mut StudyNode, formats: &mut bool) {
    let id = ui.make_persistent_id(("exp_study", st.slot, &st.source_uid));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let title = format!(
        "📁 {}",
        [st.date.trimmed(), st.description.trimmed()]
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("  ")
    );
    let counts = format!("{} series, {} object(s)", st.series.len(), st.objects.len());
    state
        .show_header(ui, |ui| {
            if let Some(on) = tri_checkbox(ui, st.all_selected()) {
                st.set_all(on);
            }
            ui.label(if title.trim() == "📁" {
                "📁 (study)".to_string()
            } else {
                title
            });
            ui.weak(counts);
        })
        .body(|ui| {
            egui::Grid::new(id.with("f"))
                .num_columns(2)
                .spacing([8.0, 3.0])
                .show(ui, |ui| {
                    uid_row(ui, "StudyInstanceUID", &mut st.uid);
                    text_row(ui, "StudyDescription", &mut st.description, NAME_W);
                    text_row(ui, "StudyID", &mut st.id, 60.0);
                    text_row(ui, "StudyDate", &mut st.date, 90.0);
                });
            ui.add_space(3.0);

            // 4D acquisitions first, so their phases stay together.
            let groups = st.groups.clone();
            for (gi, g) in groups.iter().enumerate() {
                group_node(ui, st, gi, g);
            }
            for si in 0..st.series.len() {
                if st.series[si].fourd.is_some() {
                    continue;
                }
                series_node(ui, &mut st.series[si]);
            }
            for ob in &mut st.objects {
                object_node(ui, ob, formats);
            }
        });
}

/// A 4D acquisition. Ticking it takes every phase, which is the only way the
/// export is still one acquisition on the other side.
fn group_node(ui: &mut egui::Ui, st: &mut StudyNode, gi: usize, g: &GroupNode) {
    let id = ui.make_persistent_id(("exp_4d", st.slot, &st.source_uid, gi));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let picked = g.members.iter().filter(|i| st.series[**i].selected).count();
    let tri = match (picked, g.members.len()) {
        (0, _) => Some(false),
        (p, n) if p == n => Some(true),
        _ => None,
    };
    let title = format!("📈 {}", g.name);
    let detail = format!(
        "{} series, {} phase(s){}",
        g.members.len(),
        g.phases,
        if tri.is_none() {
            " - a partial export is not a 4D acquisition"
        } else {
            ""
        }
    );
    state
        .show_header(ui, |ui| {
            if let Some(on) = tri_checkbox(ui, tri) {
                for i in &g.members {
                    st.series[*i].selected = on;
                }
            }
            ui.label(title);
            if tri.is_none() {
                ui.colored_label(ui.visuals().warn_fg_color, detail);
            } else {
                ui.weak(detail);
            }
        })
        .body(|ui| {
            for i in &g.members {
                series_node(ui, &mut st.series[*i]);
            }
        });
}

fn series_node(ui: &mut egui::Ui, s: &mut SeriesNode) {
    let id = ui.make_persistent_id(("exp_ser", s.slot, &s.source_uid));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    let phase = s
        .fourd
        .as_ref()
        .map(|(_, l)| format!("  [{l}]"))
        .unwrap_or_default();
    let title = format!(
        "{} {}{phase}",
        if s.modality.is_empty() {
            "IM"
        } else {
            &s.modality
        },
        s.description.trimmed()
    );
    let detail = if s.n_files == 0 {
        "rendered from the loaded voxels".to_string()
    } else {
        format!("{} file(s)", s.n_files)
    };
    state
        .show_header(ui, |ui| {
            ui.checkbox(&mut s.selected, "");
            ui.label(title);
            ui.weak(detail);
        })
        .body(|ui| {
            egui::Grid::new(id.with("f"))
                .num_columns(2)
                .spacing([8.0, 3.0])
                .show(ui, |ui| {
                    uid_row(ui, "SeriesInstanceUID", &mut s.uid);
                    uid_row(ui, "FrameOfReferenceUID", &mut s.for_uid);
                    text_row(ui, "SeriesDescription", &mut s.description, NAME_W);
                    text_row(ui, "SeriesNumber", &mut s.number, 60.0);
                });
        });
}

fn object_node(ui: &mut egui::Ui, o: &mut ObjNode, formats: &mut bool) {
    let id = ui.make_persistent_id(("exp_obj", o.slot, o.kind as u8, o.index));
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    let title = format!("{} {}", o.kind.glyph(), o.label);
    let detail = o.detail.clone();
    let structures = o.kind.is_structures();
    state
        .show_header(ui, |ui| {
            ui.checkbox(&mut o.selected, "");
            ui.label(title);
            ui.weak(detail);
            if structures {
                ui.separator();
                *formats |= ui
                    .radio_value(&mut o.format, StructFormat::RtStruct, "RTSTRUCT")
                    .on_hover_text("Contours (RT Structure Set Storage)")
                    .clicked();
                *formats |= ui
                    .radio_value(&mut o.format, StructFormat::Seg, "SEG")
                    .on_hover_text(
                        "Binary masks on the image lattice (Segmentation Storage). \
                         Contours are rasterised, masks are written as they are.",
                    )
                    .clicked();
            }
        })
        .body(|ui| {
            egui::Grid::new(id.with("f"))
                .num_columns(2)
                .spacing([8.0, 3.0])
                .show(ui, |ui| {
                    uid_row(ui, "SOPInstanceUID", &mut o.sop_uid);
                    uid_row(ui, "SeriesInstanceUID", &mut o.series_uid);
                    let label = match o.kind {
                        ObjKind::Plan => "RTPlanName",
                        ObjKind::Structures | ObjKind::Segmentation => "StructureSetLabel",
                        ObjKind::Dose => "SeriesDescription",
                    };
                    text_row(ui, label, &mut o.description, NAME_W);
                    text_row(ui, "SeriesNumber", &mut o.number, 60.0);
                });
            if !o.referenced_series_uid.is_empty() {
                ui.weak(format!("drawn on image series {}", o.referenced_series_uid));
            }
        });
}

/// The run-wide tags the tree does not own.
fn common_tags(ui: &mut egui::Ui, plan: &mut ExportPlan) {
    let id = ui.make_persistent_id("exp_common");
    let state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    let mut reset = false;
    state
        .show_header(ui, |ui| {
            ui.label(egui::RichText::new("Common tags").strong());
            ui.weak("written into every exported object");
            if ui
                .small_button("↺ all")
                .on_hover_text("Restore every value to the study's own")
                .clicked()
            {
                reset = true;
            }
        })
        .body(|ui| {
            ui.weak(
                "Patient, study and series identity is edited in the tree above. On a copied \
                 image series only the rows you change here are applied, so the scanner's own \
                 equipment tags are not overwritten.",
            );
            egui::Grid::new(id.with("g"))
                .num_columns(2)
                .striped(true)
                .spacing([10.0, 3.0])
                .show(ui, |ui| {
                    for f in plan
                        .params
                        .fields
                        .iter_mut()
                        .filter(|f| !crate::export::PER_NODE_TAGS.contains(&f.tag))
                    {
                        let label =
                            format!("({:04X},{:04X}) {}", f.tag.group(), f.tag.element(), f.name);
                        ui.checkbox(&mut f.enabled, label).on_hover_text(format!(
                            "VR {} - unchecked: the tag is left out of the exported files",
                            f.vr
                        ));
                        ui.horizontal(|ui| {
                            ui.add_enabled(
                                f.enabled,
                                egui::TextEdit::singleline(&mut f.value)
                                    .desired_width(NAME_W)
                                    .hint_text("(empty)"),
                            );
                            if f.value != f.suggested && ui.small_button("↺").clicked() {
                                f.value = f.suggested.clone();
                            }
                        });
                        ui.end_row();
                    }
                });
        });
    if reset {
        for f in &mut plan.params.fields {
            f.value = f.suggested.clone();
            f.enabled = true;
        }
    }
}

// -- field rows -------------------------------------------------------------

fn label_of(name: &Field, id: &Field, key: &str) -> String {
    match (name.trimmed(), id.trimmed()) {
        ("", "") => key.to_string(),
        ("", i) => i.to_string(),
        (n, "") => n.to_string(),
        (n, i) => format!("{n}  ({i})"),
    }
}

fn text_row(ui: &mut egui::Ui, name: &str, f: &mut Field, width: f32) {
    ui.label(name);
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut f.value)
                .desired_width(width)
                .hint_text("(empty)"),
        );
        revert_button(ui, f);
    });
    ui.end_row();
}

/// An identifier row: the value, a way back to the original, and a way to a
/// brand new one - per field, because "new UIDs" is sometimes true of one
/// series and not of the study around it.
fn uid_row(ui: &mut egui::Ui, name: &str, f: &mut Field) {
    ui.label(name);
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut f.value)
                .desired_width(UID_W)
                .font(egui::TextStyle::Monospace)
                .hint_text("(generated)"),
        );
        revert_button(ui, f);
        if let Some(fresh) = f.fresh.clone() {
            if ui
                .add_enabled(f.value != fresh, egui::Button::new("⟳").small())
                .on_hover_text("Replace with a newly generated UID")
                .clicked()
            {
                f.value = fresh;
            }
        }
        if f.is_new() && !f.original.is_empty() {
            ui.weak("new");
        }
    });
    ui.end_row();
}

fn revert_button(ui: &mut egui::Ui, f: &mut Field) {
    if f.value != f.original
        && ui
            .add(egui::Button::new("↺").small())
            .on_hover_text(format!(
                "Back to what the data says: “{}”",
                if f.original.is_empty() {
                    "(empty)"
                } else {
                    &f.original
                }
            ))
            .clicked()
    {
        f.value = f.original.clone();
    }
}
