//! *Tools ▶ 🏥 PACS* on a paired server: the remote half of the window.
//!
//! The PACS window has a row of sources at the top: this station's own
//! archive (drawn by `pacs_win.rs`, unchanged) and every server this device
//! has paired with. Picking a server shows its archive here, in the two
//! ways of working:
//!
//! * **Studies** (the mirror): the server's patients and studies, each
//!   marked with what this device holds of it. *Pull* copies a study into
//!   the local mirror ([`crate::pacs::mirror`]), *Load into workspace* pulls
//!   if needed and loads the mirrored folder with the ordinary loader, *Send
//!   workspace* gives back the structure sets and segmentations drawn on it
//!   (or keeps them in the outbox while the server cannot be reached),
//!   *Sync* does both directions for everything mirrored.
//! * **Tasks**: the workflows the server offers, bound to its studies, run
//!   there; the queue with progress, log and cancel; *Pull results* for a
//!   finished task.
//!
//! *Add server* pairs with a new one: the connection line the server's
//! operator copied (or an address), a first look at its certificate, the
//! pairing code, done. Every call to a server runs as one background job
//! ([`ViewerApp::pacs_remote_job`]); nothing here blocks a frame.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::pacs::client::{failure_of, Failure, Remote, Trust};
use crate::pacs::mirror::{self, Mirror, PullSummary, Sent, SyncSummary};
use crate::pacs::protocol::*;
use crate::pacs::servers::{self, ServerEntry, Servers};
use crate::pacs::ConnectionLine;

use super::pacs_win::PacsWindow;
use super::*;

/// Which archive the PACS window shows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) enum Source {
    /// This station's own archive.
    #[default]
    Local,
    /// A paired server, by its id.
    Remote(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Tab {
    #[default]
    Studies,
    Tasks,
}

/// What the window shows of one server.
#[derive(Default)]
pub(super) struct RemoteView {
    pub server_id: String,
    pub tab: Tab,
    /// The server's listing: live, or as last seen when `offline`.
    pub listing: Option<Vec<RemotePatient>>,
    pub offline: bool,
    /// Something only pairing again mends: a changed certificate, a
    /// revoked token.
    pub problem: Option<String>,
    /// The role the server gives this station, as it last said.
    pub role: Option<Role>,
    /// Studies held in the mirror.
    pub local: BTreeSet<String>,
    pub pending: usize,
    pub mirror_bytes: u64,
    pub expanded: Option<usize>,
    pub selected: Option<(usize, Option<usize>)>,
    pub status: Option<String>,
    /// Asked once more before the server is forgotten.
    pub confirm_forget: bool,
    // ---- tasks ----
    pub workflows: Option<Vec<WorkflowInfo>>,
    pub tasks: Vec<TaskInfo>,
    pub form: TaskForm,
    /// The task whose details are open, with its log.
    pub watched: Option<TaskInfo>,
    /// `ctx` time of the last automatic refresh of the task list.
    pub polled_at: f64,
}

/// The *New task* form.
#[derive(Clone, Debug)]
pub(super) struct TaskForm {
    pub workflow: usize,
    /// Input node -> (patient key, study uid); an empty study is the whole
    /// patient.
    pub bindings: BTreeMap<u32, (String, String)>,
    pub title: String,
    pub file_results: bool,
}

impl Default for TaskForm {
    fn default() -> Self {
        TaskForm {
            workflow: 0,
            bindings: BTreeMap::new(),
            title: String::new(),
            file_results: true,
        }
    }
}

/// *Add server*.
#[derive(Default)]
pub(super) struct PairDialog {
    /// The connection line or an address.
    pub line: String,
    pub code: String,
    pub device_name: String,
    /// Verify through the system's certificates (a server behind a reverse
    /// proxy with a public certificate) instead of pinning.
    pub system_trust: bool,
    /// What the first look found: the server and its certificate.
    pub probed: Option<(ServerInfo, String)>,
    /// The person compared the fingerprint with the server's window.
    pub compared: bool,
    pub error: Option<String>,
    /// Re-pairing this server (its entry is replaced).
    pub again: Option<String>,
}

/// What a remote job answers with.
pub(super) enum RemoteOutcome {
    Refreshed {
        server_id: String,
        listing: Option<Vec<RemotePatient>>,
        /// Why the live listing could not be had.
        failed: Option<anyhow::Error>,
        role: Option<Role>,
        /// What the mirror holds, its outbox and its size: read in the job,
        /// since a large mirror takes a moment to walk.
        local: BTreeSet<String>,
        pending: usize,
        bytes: u64,
    },
    Pulled(PullSummary, Option<(usize, PathBuf)>),
    Sent(Sent),
    Synced(SyncSummary),
    Uploaded(UploadSummary),
    RemovedLocal,
    RemovedRemote,
    Workflows(Vec<WorkflowInfo>),
    Tasks(Vec<TaskInfo>),
    Submitted(Submitted),
    Watched(TaskInfo),
    Cancelled,
    Saved(PathBuf),
    Probed(ServerInfo, String),
    Paired(ServerEntry),
}

/// A pick in the listing: a patient, or one of their studies.
type Pick = (usize, Option<usize>);

/// What a click in the window asks for; carried out after the window is
/// drawn, with the app at hand.
pub(super) enum Action {
    Select(Source),
    AddServer,
    Refresh,
    Sync,
    Pull(Pick),
    Load(usize, Pick),
    Send(usize),
    UploadFolder,
    RemoveLocal(Pick),
    RemoveRemote(Pick),
    PairAgain,
    Forget,
    Tab(Tab),
    LoadWorkflows,
    RefreshTasks,
    Submit,
    Cancel(String),
    Watch(String),
    PullTask(String),
    SaveTaskFile(String, String),
    Probe,
    Pair,
    ClosePairing,
}

/// What the drawing needs to know besides the window itself.
pub(super) struct Ctx<'a> {
    pub servers: &'a Servers,
    pub busy: bool,
    pub progress: Option<&'a Progress>,
    pub loaded: [bool; MAX_WORKSPACES],
    pub targets: Vec<usize>,
}

/// The device name a station pairs under by default.
fn default_device_name() -> String {
    crate::pacs::this_device_name()
}

/// Bytes as a person reads them.
fn size_text(b: u64) -> String {
    let mb = b as f64 / (1024.0 * 1024.0);
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

/// The row of sources at the top of the PACS window. Returns the actions
/// its buttons asked for.
pub(super) fn source_row(ui: &mut egui::Ui, w: &PacsWindow, c: &Ctx, actions: &mut Vec<Action>) {
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new("Archive").strong());
        if ui
            .add(egui::Button::selectable(
                w.source == Source::Local,
                "🏥 This station",
            ))
            .on_hover_text("The archive on this computer")
            .clicked()
        {
            actions.push(Action::Select(Source::Local));
        }
        for s in &c.servers.servers {
            let on = w.source == Source::Remote(s.server_id.clone());
            if ui
                .add(egui::Button::selectable(on, format!("🔗 {}", s.name)))
                .on_hover_text(format!(
                    "{}\nrole: {}\npaired {}",
                    s.label(),
                    s.role.describe(),
                    s.paired
                ))
                .clicked()
            {
                actions.push(Action::Select(Source::Remote(s.server_id.clone())));
            }
        }
        if ui
            .add_enabled(!c.busy, egui::Button::new("➕ Add server"))
            .on_hover_text(
                "Pair this station with a PACS server: paste the connection line its \
                 operator copied, check its certificate, type the pairing code",
            )
            .clicked()
        {
            actions.push(Action::AddServer);
        }
    });
}

/// The pairing dialog, drawn in place of the window's body.
pub(super) fn pairing_ui(
    ui: &mut egui::Ui,
    d: &mut PairDialog,
    c: &Ctx,
    actions: &mut Vec<Action>,
) {
    ui.heading(if d.again.is_some() {
        "🔒 Pair again"
    } else {
        "➕ Add a PACS server"
    });
    ui.label(
        "The server's operator copies its connection details in Settings > PACS server \
         (or runs rds-pacs pair) and gives you the line and a pairing code.",
    );
    ui.add_space(4.0);
    egui::Grid::new("pacs_pair")
        .num_columns(2)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            ui.label("Connection line or address");
            let mut changed = false;
            ui.horizontal(|ui| {
                changed |= ui
                    .add(
                        egui::TextEdit::singleline(&mut d.line)
                            .desired_width(420.0)
                            .hint_text(
                                "rds-pacs://192.168.1.20:11443/#sha256=... or pacs.example.org",
                            ),
                    )
                    .changed();
                if let Some(text) = system_paste_button(ui) {
                    d.line = text;
                    changed = true;
                }
            });
            if changed {
                d.probed = None;
                d.compared = false;
                d.error = None;
                if let Ok(l) = ConnectionLine::parse(&d.line) {
                    if let Some(code) = l.code {
                        d.code = code;
                    }
                }
            }
            ui.end_row();
            ui.label("");
            ui.checkbox(
                &mut d.system_trust,
                "The server has a certificate the system trusts (behind a reverse proxy \
                 with a public certificate)",
            );
            ui.end_row();
        });
    ui.horizontal(|ui| {
        if ui
            .add_enabled(
                !c.busy && !d.line.trim().is_empty(),
                egui::Button::new("🔍 Look at the server"),
            )
            .on_hover_text("Connect without sending anything and show who answers")
            .clicked()
        {
            actions.push(Action::Probe);
        }
        if ui.button("Cancel").clicked() {
            actions.push(Action::ClosePairing);
        }
    });
    if let Some(p) = c.progress {
        progress_row(ui, p);
    }
    if let Some(e) = &d.error {
        ui.colored_label(ui.visuals().error_fg_color, e);
    }
    let Some((info, fp)) = d.probed.clone() else {
        return;
    };
    ui.separator();
    ui.label(format!(
        "{} answers: Rust DICOM Station {}, protocol {}{}",
        info.name,
        info.version,
        info.api_version,
        if info.tasks { ", runs tasks" } else { "" }
    ));
    let line_fp = ConnectionLine::parse(&d.line)
        .ok()
        .and_then(|l| l.fingerprint);
    let trusted = if d.system_trust {
        ui.weak(
            "The certificate is checked against the system's certificates when you pair; \
             no fingerprint is pinned.",
        );
        true
    } else {
        ui.label("Its certificate:");
        ui.label(
            egui::RichText::new(crate::pacs::show_fingerprint(&fp))
                .monospace()
                .strong(),
        );
        match &line_fp {
            Some(want) if *want == fp => {
                ui.colored_label(
                    egui::Color32::from_rgb(60, 160, 80),
                    "✔ the same as in the connection line",
                );
                true
            }
            Some(_) => {
                ui.colored_label(
                    ui.visuals().error_fg_color,
                    "⚠ NOT the certificate in the connection line. Do not pair: ask the \
                     operator whether the server's certificate was renewed.",
                );
                false
            }
            None => {
                ui.checkbox(
                    &mut d.compared,
                    "I compared it with the fingerprint the server's window shows, and it \
                     is the same",
                );
                d.compared
            }
        }
    };
    ui.add_space(4.0);
    egui::Grid::new("pacs_pair_code")
        .num_columns(2)
        .spacing([8.0, 6.0])
        .show(ui, |ui| {
            ui.label("Pairing code");
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut d.code)
                        .desired_width(140.0)
                        .hint_text("K7QM-3TXA"),
                );
                if let Some(text) = system_paste_button(ui) {
                    // A whole invitation line pasted here: take its code.
                    d.code = ConnectionLine::parse(&text)
                        .ok()
                        .and_then(|l| l.code)
                        .unwrap_or(text);
                }
            });
            ui.end_row();
            ui.label("This station's name");
            if d.device_name.is_empty() {
                d.device_name = default_device_name();
            }
            ui.add(egui::TextEdit::singleline(&mut d.device_name).desired_width(240.0));
            ui.end_row();
        });
    if ui
        .add_enabled(
            !c.busy && trusted && !d.code.trim().is_empty(),
            egui::Button::new("🔒 Pair"),
        )
        .on_hover_text("Trade the code for this station's own key to the server")
        .clicked()
    {
        actions.push(Action::Pair);
    }
}

/// One study's mark: what this device holds of it.
fn held_mark(v: &RemoteView, uid: &str) -> &'static str {
    if v.local.contains(uid) {
        "✔ "
    } else {
        ""
    }
}

/// The body of the window for a paired server.
pub(super) fn remote_ui(
    ui: &mut egui::Ui,
    v: &mut RemoteView,
    entry: &ServerEntry,
    c: &Ctx,
    actions: &mut Vec<Action>,
) {
    let role = v.role.unwrap_or(entry.role);
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(format!("🔗 {}", entry.label())).strong());
        ui.weak(format!("· role {}", role.label()))
            .on_hover_text(role.describe());
        if v.pending > 0 {
            ui.weak(format!("· {} file(s) in the outbox", v.pending));
        }
        ui.weak(format!("· this device holds {}", size_text(v.mirror_bytes)));
    });
    ui.horizontal_wrapped(|ui| {
        if enabled_tip_button(ui, !c.busy, "⟲ Refresh", "Read the server's archive again") {
            actions.push(Action::Refresh);
        }
        if enabled_tip_button(
            ui,
            !c.busy && role.allows(Role::Edit),
            "⟲ Sync",
            "Both directions for every study held here: send what waits in the outbox, \
             fetch what others filed meanwhile, send what the server lacks",
        ) {
            actions.push(Action::Sync);
        }
        if enabled_tip_button(
            ui,
            !c.busy && role.allows(Role::Edit),
            "📤 Upload folder",
            "Send every DICOM file of a folder of this computer into the server's archive",
        ) {
            actions.push(Action::UploadFolder);
        }
        if enabled_tip_button(
            ui,
            !c.busy,
            "🔒 Pair again",
            "A new pairing code and, if the server's certificate was renewed, its new \
             fingerprint",
        ) {
            actions.push(Action::PairAgain);
        }
        let forget = if v.confirm_forget {
            "✖ Really forget it?"
        } else {
            "✖ Forget server"
        };
        if enabled_tip_button(
            ui,
            !c.busy,
            forget,
            "Remove the server from this station. What was pulled stays in the mirror \
             folder until removed by hand.",
        ) {
            actions.push(Action::Forget);
        }
    });
    if let Some(p) = &v.problem {
        ui.colored_label(ui.visuals().error_fg_color, p);
    }
    if let Some(p) = c.progress {
        progress_row(ui, p);
    }
    if let Some(s) = &v.status {
        ui.weak(s);
    }
    ui.separator();
    ui.horizontal(|ui| {
        if ui
            .add(egui::Button::selectable(v.tab == Tab::Studies, "Studies"))
            .clicked()
        {
            actions.push(Action::Tab(Tab::Studies));
        }
        let tasks = ui.add_enabled(
            role.allows(Role::Run),
            egui::Button::selectable(v.tab == Tab::Tasks, "Tasks"),
        );
        if tasks
            .on_hover_text("Workflows run on the server, on its studies")
            .on_disabled_hover_text("This station is paired without the run role")
            .clicked()
        {
            actions.push(Action::Tab(Tab::Tasks));
        }
    });
    ui.separator();
    match v.tab {
        Tab::Studies => studies_ui(ui, v, role, c, actions),
        Tab::Tasks => tasks_ui(ui, v, c, actions),
    }
}

fn studies_ui(
    ui: &mut egui::Ui,
    v: &mut RemoteView,
    role: Role,
    c: &Ctx,
    actions: &mut Vec<Action>,
) {
    let Some(patients) = &v.listing else {
        ui.weak("reading");
        return;
    };
    if v.offline {
        ui.weak("The server cannot be reached: its archive as last seen. What is held here works as usual.");
    }
    if patients.is_empty() {
        ui.weak("The server's archive is empty.");
    }
    let mut expand: Option<Option<usize>> = None;
    let mut select: Option<Pick> = None;
    egui::ScrollArea::vertical()
        .max_height(300.0)
        .id_salt("pacs_remote_list")
        .show(ui, |ui| {
            for (pi, p) in patients.iter().enumerate() {
                let open = v.expanded == Some(pi);
                let held = p
                    .studies
                    .iter()
                    .filter(|s| v.local.contains(&s.study_uid))
                    .count();
                ui.horizontal(|ui| {
                    if ui.small_button(if open { "▼" } else { "▶" }).clicked() {
                        expand = Some(if open { None } else { Some(pi) });
                    }
                    let resp = ui.add(
                        egui::Button::selectable(
                            v.selected == Some((pi, None)),
                            format!(
                                "{}   {} study(ies) · {} file(s){}",
                                p.title(),
                                p.studies.len(),
                                p.files(),
                                if held > 0 {
                                    format!(" · {held} held here")
                                } else {
                                    String::new()
                                }
                            ),
                        )
                        .wrap(),
                    );
                    if resp.clicked() {
                        select = Some((pi, None));
                        expand = Some(Some(pi));
                    }
                    resp.context_menu(|ui| {
                        if ui.button("🗑 Remove the local copy").clicked() {
                            actions.push(Action::RemoveLocal((pi, None)));
                            ui.close();
                        }
                        if role.allows(Role::Admin)
                            && ui.button("🗑 Remove this patient from the server").clicked()
                        {
                            actions.push(Action::RemoveRemote((pi, None)));
                            ui.close();
                        }
                    });
                });
                if !open {
                    continue;
                }
                for (si, st) in p.studies.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.add_space(24.0);
                        let resp = ui.add(
                            egui::Button::selectable(
                                v.selected == Some((pi, Some(si))),
                                format!("{}{}", held_mark(v, &st.study_uid), st.describe()),
                            )
                            .wrap(),
                        );
                        if resp.clicked() {
                            select = Some((pi, Some(si)));
                        }
                        let resp = resp.on_hover_text(format!(
                            "Study UID …{}{}",
                            tail(&st.study_uid),
                            if v.local.contains(&st.study_uid) {
                                "\n✔ held on this device"
                            } else {
                                "\non the server only"
                            }
                        ));
                        resp.context_menu(|ui| {
                            if ui.button("🗑 Remove the local copy").clicked() {
                                actions.push(Action::RemoveLocal((pi, Some(si))));
                                ui.close();
                            }
                            if role.allows(Role::Admin)
                                && ui.button("🗑 Remove this study from the server").clicked()
                            {
                                actions.push(Action::RemoveRemote((pi, Some(si))));
                                ui.close();
                            }
                        });
                    });
                }
            }
        });
    if let Some(e) = expand {
        v.expanded = e;
    }
    if let Some(s) = select {
        v.selected = Some(s);
    }
    ui.separator();
    let picked = v.selected.filter(|(pi, si)| {
        patients
            .get(*pi)
            .is_some_and(|p| si.is_none_or(|si| si < p.studies.len()))
    });
    ui.horizontal_wrapped(|ui| {
        if enabled_tip_button(
            ui,
            !c.busy && picked.is_some() && !v.offline,
            "📥 Pull",
            "Copy the selection into this device's mirror (only what it does not hold yet)",
        ) {
            if let Some(p) = picked {
                actions.push(Action::Pull(p));
            }
        }
        for slot in c.targets.iter().copied() {
            if ui
                .add_enabled(
                    !c.busy && picked.is_some(),
                    egui::Button::new(format!("📩 Load into workspace {}", SLOT_NAMES[slot])),
                )
                .on_hover_text(
                    "Pull what is missing, then read the selection into this workspace \
                     from the mirror, like any folder",
                )
                .clicked()
            {
                if let Some(p) = picked {
                    actions.push(Action::Load(slot, p));
                }
            }
        }
    });
    ui.horizontal_wrapped(|ui| {
        for slot in (0..MAX_WORKSPACES).filter(|s| c.loaded[*s]) {
            if ui
                .add_enabled(
                    !c.busy && role.allows(Role::Edit),
                    egui::Button::new(format!("📤 Send workspace {}", SLOT_NAMES[slot])),
                )
                .on_hover_text(
                    "Give the server this workspace's structure sets and segmentation \
                     series, filed under the study they were drawn on (new SOP Instance \
                     UIDs; images are never re-sent). Kept in the outbox while the server \
                     cannot be reached.",
                )
                .on_disabled_hover_text("This station is paired with the view role only")
                .clicked()
            {
                actions.push(Action::Send(slot));
            }
        }
    });
}

fn tasks_ui(ui: &mut egui::Ui, v: &mut RemoteView, c: &Ctx, actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Queue").strong());
        if ui
            .add_enabled(!c.busy, egui::Button::new("⟲"))
            .on_hover_text("Read the queue again")
            .clicked()
        {
            actions.push(Action::RefreshTasks);
        }
    });
    if v.tasks.is_empty() {
        ui.weak("No task has been handed in yet.");
    }
    egui::ScrollArea::vertical()
        .max_height(170.0)
        .id_salt("pacs_tasks")
        .show(ui, |ui| {
            egui::Grid::new("pacs_task_grid")
                .striped(true)
                .num_columns(5)
                .show(ui, |ui| {
                    for t in &v.tasks {
                        ui.label(&t.title).on_hover_text(format!(
                            "{}\nby {} at {}",
                            t.workflow, t.client, t.submitted
                        ));
                        match t.state {
                            TaskState::Running => {
                                ui.add(
                                    egui::ProgressBar::new(t.progress)
                                        .desired_width(120.0)
                                        .show_percentage(),
                                )
                                .on_hover_text(&t.message);
                            }
                            s => {
                                ui.label(s.label());
                            }
                        }
                        ui.weak(&t.submitted);
                        if ui.small_button("Details").clicked() {
                            actions.push(Action::Watch(t.id.clone()));
                        }
                        ui.horizontal(|ui| {
                            if !t.state.finished() && ui.small_button("✖ Cancel").clicked() {
                                actions.push(Action::Cancel(t.id.clone()));
                            }
                            if t.state == TaskState::Done
                                && !t.filed.is_empty()
                                && ui
                                    .add_enabled(
                                        !c.busy,
                                        egui::Button::new("📥 Pull results").small(),
                                    )
                                    .clicked()
                            {
                                actions.push(Action::PullTask(t.id.clone()));
                            }
                        });
                        ui.end_row();
                    }
                });
        });
    if let Some(t) = &v.watched {
        ui.separator();
        ui.label(egui::RichText::new(format!("{} - {}", t.title, t.state.label())).strong());
        if let Some(e) = &t.error {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        if !t.filed.is_empty() {
            ui.weak(format!("filed into {} study(ies)", t.filed.len()));
        }
        egui::ScrollArea::vertical()
            .max_height(140.0)
            .id_salt("pacs_task_log")
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for l in &t.log {
                    ui.monospace(l);
                }
            });
        if !t.files.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.weak("Files:");
                for f in &t.files {
                    if ui
                        .small_button(format!("💾 {f}"))
                        .on_hover_text("Save this file of the task's on this computer")
                        .clicked()
                    {
                        actions.push(Action::SaveTaskFile(t.id.clone(), f.clone()));
                    }
                }
            });
        }
    }
    ui.separator();
    ui.label(egui::RichText::new("New task").strong());
    let Some(wfs) = &v.workflows else {
        ui.weak("reading what the server offers");
        return;
    };
    if wfs.is_empty() {
        ui.weak("The server offers no workflow.");
        return;
    }
    let listing = v.listing.clone().unwrap_or_default();
    let f = &mut v.form;
    f.workflow = f.workflow.min(wfs.len() - 1);
    let before = f.workflow;
    egui::ComboBox::from_id_salt("pacs_wf")
        .width(360.0)
        .selected_text(&wfs[f.workflow].name)
        .show_ui(ui, |ui| {
            for (i, w) in wfs.iter().enumerate() {
                ui.selectable_value(&mut f.workflow, i, format!("{} ({})", w.name, w.source));
            }
        });
    if f.workflow != before {
        f.bindings.clear();
    }
    let wf = &wfs[f.workflow];
    if !wf.description.is_empty() {
        ui.weak(&wf.description);
    }
    ui.weak(format!("Steps: {}", wf.steps.join(" > ")));
    for p in &wf.problems {
        ui.colored_label(ui.visuals().warn_fg_color, format!("⚠ {p}"));
    }
    egui::Grid::new("pacs_bind")
        .num_columns(2)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            for input in &wf.inputs {
                ui.label(&input.title);
                let cur = f.bindings.get(&input.node).cloned().unwrap_or_default();
                let shown = listing
                    .iter()
                    .find(|p| p.key == cur.0)
                    .map(|p| match p.studies.iter().find(|s| s.study_uid == cur.1) {
                        Some(s) => format!("{} · {}", p.title(), s.describe()),
                        None => format!("{} · all studies", p.title()),
                    })
                    .unwrap_or_else(|| "choose".into());
                egui::ComboBox::from_id_salt(("pacs_bind", input.node))
                    .width(420.0)
                    .selected_text(shown)
                    .show_ui(ui, |ui| {
                        for p in &listing {
                            if input.kind != InputKind::Archive {
                                let v = (p.key.clone(), String::new());
                                let label = if input.kind == InputKind::Batch {
                                    format!("{} (each study a case)", p.title())
                                } else {
                                    format!("{} · all studies", p.title())
                                };
                                if ui.selectable_label(cur == v, label).clicked() {
                                    f.bindings.insert(input.node, v);
                                }
                            }
                            if input.kind == InputKind::Batch {
                                continue;
                            }
                            for s in &p.studies {
                                let v = (p.key.clone(), s.study_uid.clone());
                                if ui
                                    .selectable_label(
                                        cur == v,
                                        format!("{} · {}", p.title(), s.describe()),
                                    )
                                    .clicked()
                                {
                                    f.bindings.insert(input.node, v);
                                }
                            }
                        }
                    });
                ui.end_row();
            }
            ui.label("Title");
            ui.add(
                egui::TextEdit::singleline(&mut f.title)
                    .desired_width(260.0)
                    .hint_text(&wf.name),
            );
            ui.end_row();
        });
    ui.checkbox(
        &mut f.file_results,
        "File the results into the server's archive (so they can be pulled)",
    );
    let complete = wf.inputs.iter().all(|i| f.bindings.contains_key(&i.node));
    if ui
        .add_enabled(
            !c.busy && complete && wf.problems.is_empty(),
            egui::Button::new("▶ Run on the server"),
        )
        .on_hover_text("Hand the task in; the server runs one task at a time, in order")
        .on_disabled_hover_text("Choose a study for every input")
        .clicked()
    {
        actions.push(Action::Submit);
    }
}

// ---- the jobs -----------------------------------------------------------------

impl ViewerApp {
    /// The paired servers, read again from their file.
    pub(super) fn reload_servers(&mut self) -> Servers {
        Servers::load(&servers::default_path()).unwrap_or_default()
    }

    /// The entry of the server the window shows.
    fn current_entry(&self) -> Option<ServerEntry> {
        let w = self.pacs.as_ref()?;
        let Source::Remote(id) = &w.source else {
            return None;
        };
        w.servers.get(id).cloned()
    }

    /// Run `work` against the window's server on a background thread.
    fn remote_job(
        &mut self,
        what: &str,
        work: impl FnOnce(&Remote, &Mirror, &Progress) -> anyhow::Result<RemoteOutcome> + Send + 'static,
    ) {
        if self.pacs_remote_job.is_some() {
            return;
        }
        let Some(entry) = self.current_entry() else {
            return;
        };
        let progress = Arc::new(Progress::default());
        progress.set(what);
        self.pacs_remote_job = Some(Job::spawn(progress, move |p| {
            let remote = Remote::for_server(&entry)?;
            let mirror = Mirror::for_server(&entry.server_id);
            work(&remote, &mirror, p)
        }));
    }

    /// Read the server's listing (or the cached one when it cannot be
    /// reached) and what the mirror holds.
    pub(super) fn remote_refresh(&mut self) {
        let id = self
            .current_entry()
            .map(|e| e.server_id)
            .unwrap_or_default();
        self.remote_job("Reading the server's archive", move |remote, mirror, _| {
            let (listing, failed, role) = match remote.patients() {
                Ok(l) => {
                    mirror.save_listing(&l);
                    let role = remote.whoami().ok().map(|w| w.role);
                    (Some(l), None, role)
                }
                Err(e) => (Some(mirror.cached_listing()), Some(e), None),
            };
            Ok(RemoteOutcome::Refreshed {
                server_id: id,
                listing,
                failed,
                role,
                local: mirror.local_studies(),
                pending: mirror.pending_files(),
                bytes: mirror.size_bytes(),
            })
        });
    }

    /// The patient and study a pick names in the current listing.
    fn resolve_pick(&self, pick: Pick) -> Option<(RemotePatient, Option<String>)> {
        let v = self.pacs.as_ref()?.remote.as_ref()?;
        let p = v.listing.as_ref()?.get(pick.0)?.clone();
        let study = match pick.1 {
            Some(si) => Some(p.studies.get(si)?.study_uid.clone()),
            None => None,
        };
        Some((p, study))
    }

    /// Carry out what the window's clicks asked for.
    pub(super) fn remote_actions(&mut self, actions: Vec<Action>) {
        for a in actions {
            self.remote_action(a);
        }
    }

    fn remote_action(&mut self, a: Action) {
        match a {
            Action::Select(src) => {
                let servers = self.reload_servers();
                if let Some(w) = self.pacs.as_mut() {
                    w.servers = servers;
                    w.pairing = None;
                    w.remote = match &src {
                        Source::Remote(id) => Some(RemoteView {
                            server_id: id.clone(),
                            ..Default::default()
                        }),
                        Source::Local => None,
                    };
                    w.source = src.clone();
                }
                if matches!(src, Source::Remote(_)) {
                    self.remote_refresh();
                }
            }
            Action::AddServer => {
                if let Some(w) = self.pacs.as_mut() {
                    w.pairing = Some(PairDialog {
                        device_name: default_device_name(),
                        ..Default::default()
                    });
                }
            }
            Action::ClosePairing => {
                if let Some(w) = self.pacs.as_mut() {
                    w.pairing = None;
                }
            }
            Action::PairAgain => {
                let Some(e) = self.current_entry() else {
                    return;
                };
                let line = ConnectionLine::parse(&e.url)
                    .map(|l| {
                        ConnectionLine {
                            fingerprint: None,
                            ..l
                        }
                        .format()
                    })
                    .unwrap_or(e.url.clone());
                if let Some(w) = self.pacs.as_mut() {
                    w.pairing = Some(PairDialog {
                        line,
                        device_name: e.client_name.clone(),
                        system_trust: e.fingerprint.is_empty(),
                        again: Some(e.server_id.clone()),
                        ..Default::default()
                    });
                }
            }
            Action::Forget => {
                let Some(e) = self.current_entry() else {
                    return;
                };
                let confirmed = self
                    .pacs
                    .as_ref()
                    .and_then(|w| w.remote.as_ref())
                    .is_some_and(|v| v.confirm_forget);
                if !confirmed {
                    if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
                        v.confirm_forget = true;
                    }
                    return;
                }
                let mut s = self.reload_servers();
                s.remove(&e.server_id);
                if let Err(err) = s.save(&servers::default_path()) {
                    self.error = Some(format!("PACS: {err:#}"));
                    return;
                }
                self.remote_action(Action::Select(Source::Local));
                if let Some(w) = self.pacs.as_mut() {
                    w.status = Some(format!(
                        "✔ {} forgotten; its mirror stays in {}",
                        e.name,
                        Mirror::for_server(&e.server_id).root().display()
                    ));
                }
            }
            Action::Refresh => self.remote_refresh(),
            Action::Tab(t) => {
                let load = if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
                    v.tab = t;
                    t == Tab::Tasks && v.workflows.is_none()
                } else {
                    false
                };
                if load {
                    self.remote_action(Action::LoadWorkflows);
                }
            }
            Action::LoadWorkflows => {
                self.remote_job("Reading what the server offers", |r, _, _| {
                    let wfs = r.workflows()?;
                    // The queue comes along on the next refresh.
                    Ok(RemoteOutcome::Workflows(wfs))
                });
            }
            Action::RefreshTasks => {
                self.remote_job("Reading the queue", |r, _, _| {
                    Ok(RemoteOutcome::Tasks(r.tasks()?))
                });
            }
            Action::Watch(id) => {
                self.remote_job("Reading the task", move |r, _, _| {
                    Ok(RemoteOutcome::Watched(r.task(&id, 0)?))
                });
            }
            Action::Cancel(id) => {
                self.remote_job("Cancelling", move |r, _, _| {
                    r.cancel_task(&id)?;
                    Ok(RemoteOutcome::Cancelled)
                });
            }
            Action::Submit => {
                let Some(req) = self.task_request() else {
                    return;
                };
                self.remote_job("Handing the task in", move |r, _, _| {
                    Ok(RemoteOutcome::Submitted(r.submit(&req)?))
                });
            }
            Action::PullTask(id) => {
                self.remote_job("Pulling the task's results", move |r, m, p| {
                    let t = r.task(&id, 0)?;
                    let mut sum = PullSummary::default();
                    for uid in &t.filed {
                        sum.add(&mirror::pull_study(r, m, uid, p)?);
                    }
                    Ok(RemoteOutcome::Pulled(sum, None))
                });
            }
            Action::SaveTaskFile(id, name) => {
                let file = Path::new(&name)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "task-file".into());
                self.ask_save(
                    "Save the task's file",
                    file,
                    None,
                    None,
                    move |app, dest| {
                        app.remote_job("Fetching the file", move |r, _, _| {
                            r.task_file(&id, &name, &dest)?;
                            Ok(RemoteOutcome::Saved(dest))
                        });
                    },
                );
            }
            Action::Pull(pick) => {
                let Some((patient, study)) = self.resolve_pick(pick) else {
                    return;
                };
                self.remote_job("Pulling", move |r, m, p| {
                    let sum = match study {
                        Some(uid) => mirror::pull_study(r, m, &uid, p)?,
                        None => mirror::pull_patient(r, m, &patient, p)?,
                    };
                    Ok(RemoteOutcome::Pulled(sum, None))
                });
            }
            Action::Load(slot, pick) => {
                let Some((patient, study)) = self.resolve_pick(pick) else {
                    return;
                };
                let offline = self
                    .pacs
                    .as_ref()
                    .and_then(|w| w.remote.as_ref())
                    .is_some_and(|v| v.offline);
                self.remote_job("Pulling, then loading", move |r, m, p| {
                    let uids: Vec<String> = match &study {
                        Some(u) => vec![u.clone()],
                        None => patient
                            .studies
                            .iter()
                            .map(|s| s.study_uid.clone())
                            .collect(),
                    };
                    let mut sum = PullSummary::default();
                    if !offline {
                        for u in &uids {
                            sum.add(&mirror::pull_study(r, m, u, p)?);
                        }
                    }
                    let a = m.archive();
                    let dir = match &study {
                        Some(u) => a.find_study(u),
                        // The patient's folder in the mirror: the parent
                        // of any of their studies.
                        None => uids
                            .iter()
                            .find_map(|u| a.find_study(u))
                            .and_then(|d| d.parent().map(Path::to_path_buf)),
                    }
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "this device holds none of it, and the server cannot be reached"
                        )
                    })?;
                    Ok(RemoteOutcome::Pulled(sum, Some((slot, dir))))
                });
            }
            Action::Send(slot) => {
                let Some(study) = self.slots[slot].study.as_ref() else {
                    return;
                };
                let derived = study.structure_sets.iter().any(|ss| !ss.rois.is_empty())
                    || study
                        .seg_series
                        .iter()
                        .any(|sr| sr.segs.iter().any(|s| s.count > 0));
                if !derived {
                    self.error = Some(format!(
                        "workspace {} has no structure sets or segmentations to send",
                        SLOT_NAMES[slot]
                    ));
                    return;
                }
                let study = study.clone();
                let params = dicom_export::ExportParams::for_study(&study);
                self.remote_job("Writing the derived objects", move |r, m, p| {
                    let scratch = std::env::temp_dir().join(format!(
                        "rds_send_{}_{}",
                        std::process::id(),
                        crate::pacs::unix_now()
                    ));
                    dicom_export::export_derived(&study, &scratch, &params, p)?;
                    let sent = mirror::send_folder(r, m, &scratch, p);
                    let _ = std::fs::remove_dir_all(&scratch);
                    Ok(RemoteOutcome::Sent(sent?))
                });
            }
            Action::UploadFolder => {
                self.ask_folder("Folder to send into the server's archive", |app, dir| {
                    app.remote_job("Sending the folder", move |r, m, p| {
                        let files = mirror::files_under(&dir);
                        let scratch = m.root().join(".upload");
                        Ok(RemoteOutcome::Uploaded(
                            r.upload_files(&files, &scratch, p)?,
                        ))
                    });
                });
            }
            Action::Sync => {
                self.remote_job("Syncing", |r, m, p| {
                    Ok(RemoteOutcome::Synced(mirror::sync(r, m, p)?))
                });
            }
            Action::RemoveLocal(pick) => {
                let Some((patient, study)) = self.resolve_pick(pick) else {
                    return;
                };
                self.remote_job("Removing the local copy", move |_, m, _| {
                    let uids: Vec<String> = match study {
                        Some(u) => vec![u],
                        None => patient
                            .studies
                            .iter()
                            .map(|s| s.study_uid.clone())
                            .collect(),
                    };
                    let held = m.local_studies();
                    for u in uids.iter().filter(|u| held.contains(*u)) {
                        m.remove_study(u)?;
                    }
                    Ok(RemoteOutcome::RemovedLocal)
                });
            }
            Action::RemoveRemote(pick) => {
                let Some((patient, study)) = self.resolve_pick(pick) else {
                    return;
                };
                self.remote_job("Removing from the server", move |r, _, _| {
                    match study {
                        Some(u) => r.delete_study(&u)?,
                        None => r.delete_patient(&patient.key)?,
                    }
                    Ok(RemoteOutcome::RemovedRemote)
                });
            }
            Action::Probe => self.start_probe(),
            Action::Pair => self.start_pair(),
        }
    }

    /// The form's request, or `None` when it is incomplete.
    fn task_request(&self) -> Option<TaskRequest> {
        let v = self.pacs.as_ref()?.remote.as_ref()?;
        let wf = v.workflows.as_ref()?.get(v.form.workflow)?;
        Some(TaskRequest {
            workflow: wf.name.clone(),
            bindings: v
                .form
                .bindings
                .iter()
                .filter(|(node, _)| wf.inputs.iter().any(|i| i.node == **node))
                .map(|(node, (key, uid))| Binding {
                    node: *node,
                    patient_key: key.clone(),
                    study_uid: uid.clone(),
                })
                .collect(),
            title: v.form.title.trim().to_string(),
            file_results: v.form.file_results,
            ..Default::default()
        })
    }

    fn start_probe(&mut self) {
        if self.pacs_remote_job.is_some() {
            return;
        }
        let Some(d) = self.pacs.as_ref().and_then(|w| w.pairing.as_ref()) else {
            return;
        };
        let line = match ConnectionLine::parse(&d.line) {
            Ok(l) => l,
            Err(e) => {
                if let Some(d) = self.pacs.as_mut().and_then(|w| w.pairing.as_mut()) {
                    d.error = Some(format!("{e:#}"));
                }
                return;
            }
        };
        let progress = Arc::new(Progress::default());
        progress.set(format!("Asking {}", line.authority()));
        self.pacs_remote_job = Some(Job::spawn(progress, move |_| {
            let (info, fp) = Remote::probe(&line.base_url())?;
            Ok(RemoteOutcome::Probed(info, fp))
        }));
    }

    fn start_pair(&mut self) {
        if self.pacs_remote_job.is_some() {
            return;
        }
        let Some(d) = self.pacs.as_ref().and_then(|w| w.pairing.as_ref()) else {
            return;
        };
        let Some((info, fp)) = d.probed.clone() else {
            return;
        };
        let Ok(line) = ConnectionLine::parse(&d.line) else {
            return;
        };
        let code = d.code.clone();
        let device = if d.device_name.trim().is_empty() {
            default_device_name()
        } else {
            d.device_name.trim().to_string()
        };
        let trust = if d.system_trust {
            Trust::System
        } else {
            Trust::Pinned(fp.clone())
        };
        let pinned = if d.system_trust { String::new() } else { fp };
        let progress = Arc::new(Progress::default());
        progress.set("Pairing");
        self.pacs_remote_job = Some(Job::spawn(progress, move |_| {
            let base = line.base_url();
            let anon = Remote::new(&base, trust.clone(), None)?;
            let got = anon.pair(&code, &device)?;
            if got.server_id != info.server_id {
                anyhow::bail!("the server answered as another server than it said it was");
            }
            Ok(RemoteOutcome::Paired(ServerEntry {
                server_id: got.server_id,
                name: info.name,
                url: base,
                fingerprint: pinned,
                token: got.token,
                role: got.role,
                client_name: got.client_name,
                paired: crate::pacs::stamp(),
            }))
        }));
    }

    /// A remote job finished.
    pub(super) fn on_remote_done(&mut self, r: anyhow::Result<RemoteOutcome>) {
        let mut refresh = false;
        let mut load: Option<(usize, PathBuf)> = None;
        let mut select: Option<String> = None;
        let mut reload_tasks = false;
        match r {
            Err(e) => {
                let text = format!("{e:#}");
                if let Some(d) = self.pacs.as_mut().and_then(|w| w.pairing.as_mut()) {
                    d.error = Some(match failure_of(&e) {
                        // Nothing answered at all: the two first-day causes,
                        // before anyone suspects the program.
                        Some(Failure::Unreachable(_)) => format!(
                            "{text}\n\nNothing answers at that address. Check that it is one \
                             the server's window lists now under \"How other stations reach \
                             it\" (the router may have handed the server's computer a new \
                             address since the line was copied), that both computers are on \
                             the same network, and that the server's computer lets rds-pacs \
                             through its firewall (on Windows: Settings > PACS server > Allow \
                             through the Windows firewall)."
                        ),
                        _ => text,
                    });
                    return;
                }
                if progress::is_cancellation(&e) {
                    return;
                }
                if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
                    match failure_of(&e) {
                        Some(Failure::CertificateChanged { .. })
                        | Some(Failure::Unauthorized(_)) => {
                            v.problem = Some(format!("⚠ {text}"));
                        }
                        _ => v.status = Some(format!("⚠ {text}")),
                    }
                } else {
                    self.error = Some(format!("PACS: {text}"));
                }
            }
            Ok(out) => match out {
                RemoteOutcome::Probed(info, fp) => {
                    if let Some(d) = self.pacs.as_mut().and_then(|w| w.pairing.as_mut()) {
                        d.error = None;
                        if info.api_version != API_VERSION {
                            d.error = Some(format!(
                                "the server speaks version {} of the protocol, this station \
                                 version {API_VERSION}: update the older of the two",
                                info.api_version
                            ));
                        } else {
                            d.probed = Some((info, fp));
                        }
                    }
                }
                RemoteOutcome::Paired(entry) => {
                    let mut s = self.reload_servers();
                    let id = entry.server_id.clone();
                    let name = entry.name.clone();
                    s.upsert(entry);
                    if let Err(e) = s.save(&servers::default_path()) {
                        self.error = Some(format!("PACS: {e:#}"));
                        return;
                    }
                    select = Some(id);
                    if let Some(w) = self.pacs.as_mut() {
                        w.status = Some(format!("✔ paired with {name}"));
                    }
                }
                RemoteOutcome::Refreshed {
                    server_id,
                    listing,
                    failed,
                    role,
                    local,
                    pending,
                    bytes,
                } => {
                    if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
                        if v.server_id != server_id {
                            return;
                        }
                        v.local = local;
                        v.pending = pending;
                        v.mirror_bytes = bytes;
                        v.listing = listing;
                        v.offline = failed.is_some();
                        v.problem = None;
                        if let Some(r) = role {
                            v.role = Some(r);
                        }
                        if let Some(e) = &failed {
                            match failure_of(e) {
                                Some(Failure::CertificateChanged { .. })
                                | Some(Failure::Unauthorized(_)) => {
                                    v.problem = Some(format!("⚠ {e:#}"));
                                }
                                Some(Failure::Unreachable(_)) => {}
                                _ => v.status = Some(format!("⚠ {e:#}")),
                            }
                        }
                        let p = v.listing.as_ref().map(|l| l.len()).unwrap_or(0);
                        let n: usize = v
                            .listing
                            .as_ref()
                            .map(|l| l.iter().map(|x| x.studies.len()).sum())
                            .unwrap_or(0);
                        // A refresh after a pull or a send keeps that
                        // job's line; the counts only fill an empty one.
                        if v.status
                            .as_deref()
                            .is_none_or(|s| !s.starts_with('⚠') && !s.starts_with('✔'))
                        {
                            v.status = Some(format!(
                                "{p} patient(s), {n} study(ies){}",
                                if v.offline { " (as last seen)" } else { "" }
                            ));
                        }
                        if v.selected.is_some_and(|(pi, _)| pi >= p) {
                            v.selected = None;
                        }
                        if v.tab == Tab::Tasks {
                            reload_tasks = true;
                        }
                    }
                }
                RemoteOutcome::Pulled(sum, then) => {
                    self.set_remote_status(format!("✔ {}", sum.describe()));
                    load = then;
                    refresh = true;
                }
                RemoteOutcome::Sent(s) => {
                    self.set_remote_status(format!("✔ {}", s.describe()));
                    refresh = true;
                }
                RemoteOutcome::Synced(s) => {
                    self.set_remote_status(format!("✔ synced: {}", s.describe()));
                    refresh = true;
                }
                RemoteOutcome::Uploaded(s) => {
                    self.set_remote_status(format!("✔ sent: {}", s.describe()));
                    refresh = true;
                }
                RemoteOutcome::RemovedLocal => {
                    self.set_remote_status("✔ the local copy was removed".into());
                    refresh = true;
                }
                RemoteOutcome::RemovedRemote => {
                    self.set_remote_status("✔ removed from the server".into());
                    refresh = true;
                }
                RemoteOutcome::Workflows(w) => {
                    if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
                        v.workflows = Some(w);
                    }
                    reload_tasks = true;
                }
                RemoteOutcome::Tasks(t) => {
                    let mut rewatch = None;
                    if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
                        // The open details follow their task while it moves.
                        if let Some(w) = &v.watched {
                            if let Some(x) = t.iter().find(|x| x.id == w.id) {
                                if x.log_len != w.log.len() || x.state != w.state {
                                    rewatch = Some(x.id.clone());
                                }
                            }
                        }
                        v.tasks = t;
                    }
                    if let Some(id) = rewatch {
                        self.remote_action(Action::Watch(id));
                    }
                }
                RemoteOutcome::Submitted(s) => {
                    self.set_remote_status(format!(
                        "✔ task handed in{}",
                        if s.position > 0 {
                            format!(", {} ahead of it", s.position)
                        } else {
                            String::new()
                        }
                    ));
                    reload_tasks = true;
                }
                RemoteOutcome::Watched(t) => {
                    if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
                        v.watched = Some(t);
                    }
                }
                RemoteOutcome::Cancelled => reload_tasks = true,
                RemoteOutcome::Saved(p) => {
                    self.set_remote_status(format!("✔ saved {}", p.display()));
                }
            },
        }
        if let Some(id) = select {
            self.remote_action(Action::Select(Source::Remote(id)));
            return;
        }
        if let Some((slot, dir)) = load {
            self.start_load(slot, dir);
        }
        if refresh {
            self.remote_refresh();
        } else if reload_tasks {
            self.remote_action(Action::RefreshTasks);
        }
    }

    fn set_remote_status(&mut self, s: String) {
        if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
            v.status = Some(s);
            v.confirm_forget = false;
        }
    }

    /// Is a task of the shown server queued or running (so the queue is
    /// being polled)?
    pub(super) fn remote_tasks_running(&self) -> bool {
        self.pacs
            .as_ref()
            .and_then(|w| w.remote.as_ref())
            .is_some_and(|v| v.tab == Tab::Tasks && v.tasks.iter().any(|t| !t.state.finished()))
    }

    /// Keep the queue current while a task of the shown server runs.
    pub(super) fn poll_remote_tasks(&mut self, now: f64) {
        if self.pacs_remote_job.is_some() {
            return;
        }
        let due = self
            .pacs
            .as_ref()
            .and_then(|w| w.remote.as_ref())
            .is_some_and(|v| {
                v.tab == Tab::Tasks
                    && now - v.polled_at > 2.0
                    && v.tasks.iter().any(|t| !t.state.finished())
            });
        if due {
            if let Some(v) = self.pacs.as_mut().and_then(|w| w.remote.as_mut()) {
                v.polled_at = now;
            }
            self.remote_action(Action::RefreshTasks);
        }
    }
}
