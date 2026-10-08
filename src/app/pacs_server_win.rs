//! *Settings ▶ PACS server*: the server on this machine, for its operator.
//!
//! The viewer does not run the server itself. It starts `rds-pacs serve`
//! as a process of its own (so the server keeps running when the viewer
//! closes), learns that it listens from the file the server writes, and
//! talks to it over the loopback like any client, with the local
//! operator's token the server leaves in its state folder - so this window
//! needs no pairing ([`crate::pacs::local`]).
//!
//! What it offers: whether the server is installed and running, start and
//! stop, the addresses and the certificate fingerprint other stations need
//! (and one line with all of it to copy), pairing codes, the paired
//! stations with *Revoke*, the server's settings (`pacs.toml`, applied at
//! the next start) and the tail of its audit log.

use crate::pacs::config::Config;
use crate::pacs::local::{self, Address, Firewall, Link, Paths, Running};
use crate::pacs::protocol::{ClientInfo, PairingCode, Role};
use crate::pacs::ConnectionLine;
use std::net::{IpAddr, Ipv4Addr};

use super::*;

/// The window's state.
pub(super) struct ServerWindow {
    /// `pacs.toml` as edited here.
    pub config: Config,
    /// `advertise`, as one line of names.
    pub advertise: String,
    pub config_dirty: bool,
    /// The running server, when it answers.
    pub running: Option<Running>,
    pub clients: Vec<ClientInfo>,
    pub audit: Vec<String>,
    pub code: Option<PairingCode>,
    pub code_role: Role,
    pub code_minutes: u64,
    pub message: Option<String>,
    /// `ctx` time of the last status poll.
    pub polled_at: f64,
    /// A start was asked for; the status poll waits for the server to
    /// answer.
    pub starting: bool,
    /// `ctx` time the start was asked at, to give up on a server that does
    /// not come up.
    pub start_asked: f64,
    /// Revoke asks once more.
    pub confirm_revoke: Option<String>,
    /// This machine's addresses as of the last poll (a link that comes or
    /// goes, a new lease from the router, show up within seconds).
    pub addresses: Vec<Address>,
    /// What the Windows firewall says about `rds-pacs.exe`; `None` on the
    /// other systems. Read once a while (the registry listing is long), and
    /// again after *Allow*.
    pub firewall: Option<Firewall>,
    pub firewall_read: Option<std::time::Instant>,
    /// *Allow through the Windows firewall* also on networks Windows calls
    /// public.
    pub firewall_public: bool,
}

/// What a status poll answers with.
pub(super) struct Status {
    running: Option<Running>,
    clients: Vec<ClientInfo>,
    audit: Vec<String>,
    addresses: Vec<Address>,
    firewall: Option<Firewall>,
}

/// What a job for this window answers with.
pub(super) enum ServerOutcome {
    Status(Box<Status>),
    Started,
    Stopped,
    Code(PairingCode),
    Revoked(String),
    Regenerated(String),
    FirewallAllowed,
}

impl ViewerApp {
    pub(super) fn open_pacs_server_window(&mut self) {
        if self.pacs_server.is_some() {
            return;
        }
        let (config, message) = match Config::load(&settings::pacs_config_path()) {
            Ok(c) => (c, None),
            Err(e) => (
                Config::default(),
                Some(format!("⚠ pacs.toml cannot be read: {e:#}")),
            ),
        };
        self.pacs_server = Some(ServerWindow {
            advertise: config.advertise.join(", "),
            config,
            config_dirty: false,
            running: None,
            clients: Vec::new(),
            audit: Vec::new(),
            code: None,
            code_role: Role::Edit,
            code_minutes: 10,
            message,
            polled_at: f64::NEG_INFINITY,
            starting: false,
            start_asked: 0.0,
            confirm_revoke: None,
            addresses: Vec::new(),
            firewall: None,
            firewall_read: None,
            firewall_public: false,
        });
    }

    fn server_job(
        &mut self,
        what: &str,
        work: impl FnOnce(&Progress) -> anyhow::Result<ServerOutcome> + Send + 'static,
    ) {
        if self.pacs_server_job.is_some() {
            return;
        }
        let progress = Arc::new(Progress::default());
        progress.set(what);
        self.pacs_server_job = Some(Job::spawn(progress, work));
    }

    /// Ask the server how it is, now and then while the window is open.
    pub(super) fn poll_server_status(&mut self, now: f64) {
        let Some(w) = self.pacs_server.as_mut() else {
            return;
        };
        if w.starting && w.start_asked > 0.0 && now - w.start_asked > 20.0 {
            w.starting = false;
            w.message = Some(format!(
                "⚠ the server did not come up; {} says why",
                Paths::station().log().display()
            ));
        }
        if w.starting && w.start_asked == 0.0 {
            w.start_asked = now;
        }
        let every = if w.starting { 1.0 } else { 4.0 };
        if self.pacs_server_job.is_some() || now - w.polled_at < every {
            return;
        }
        w.polled_at = now;
        let bind = w.config.bind.clone();
        let read_firewall = w
            .firewall_read
            .is_none_or(|t| t.elapsed() > std::time::Duration::from_secs(30));
        self.server_job("Asking the server", move |_| {
            let paths = Paths::station();
            let firewall = read_firewall
                .then(|| local::firewall_state(&settings::pacs_exe_path()))
                .flatten();
            match local::connect(&paths) {
                Ok((remote, run)) => {
                    let ip: IpAddr = run
                        .bind
                        .parse()
                        .unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
                    Ok(ServerOutcome::Status(Box::new(Status {
                        clients: remote.clients().unwrap_or_default(),
                        audit: remote.audit(40).unwrap_or_default(),
                        addresses: live_addresses(ip),
                        running: Some(run),
                        firewall,
                    })))
                }
                Err(_) => {
                    let ip: IpAddr = bind.parse().unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
                    Ok(ServerOutcome::Status(Box::new(Status {
                        running: None,
                        clients: local::clients_offline(&paths),
                        audit: crate::audit::tail_of(&paths.data, 40),
                        addresses: live_addresses(ip),
                        firewall,
                    })))
                }
            }
        });
    }

    pub(super) fn on_server_done(&mut self, r: anyhow::Result<ServerOutcome>) {
        let Some(w) = self.pacs_server.as_mut() else {
            return;
        };
        match r {
            Err(e) => {
                w.message = Some(format!("⚠ {e:#}"));
                w.starting = false;
            }
            Ok(ServerOutcome::Status(s)) => {
                let Status {
                    running,
                    clients,
                    audit,
                    addresses,
                    firewall,
                } = *s;
                if running.is_some() {
                    if w.starting {
                        w.message = Some("✔ the server runs".into());
                    }
                    w.starting = false;
                }
                w.running = running;
                w.clients = clients;
                w.audit = audit;
                w.addresses = addresses;
                if firewall.is_some() {
                    w.firewall = firewall;
                    w.firewall_read = Some(std::time::Instant::now());
                }
            }
            Ok(ServerOutcome::Started) => {
                w.starting = true;
                w.start_asked = 0.0;
                w.polled_at = f64::NEG_INFINITY;
                w.message = Some("Starting the server".into());
            }
            Ok(ServerOutcome::Stopped) => {
                w.running = None;
                w.code = None;
                w.polled_at = f64::NEG_INFINITY;
                w.message = Some("✔ the server was asked to stop".into());
            }
            Ok(ServerOutcome::Code(c)) => w.code = Some(c),
            Ok(ServerOutcome::Revoked(n)) => {
                w.message = Some(format!("✔ '{n}' can no longer connect"));
                w.polled_at = f64::NEG_INFINITY;
            }
            Ok(ServerOutcome::Regenerated(fp)) => {
                w.message = Some(format!(
                    "✔ a new certificate: {}. Every paired station must pair again.",
                    crate::pacs::show_fingerprint(&fp)
                ));
            }
            Ok(ServerOutcome::FirewallAllowed) => {
                w.message = Some("✔ the Windows firewall lets rds-pacs accept connections".into());
                w.firewall_read = None;
                w.polled_at = f64::NEG_INFINITY;
            }
        }
    }

    pub(super) fn pacs_server_window(&mut self, ctx: &egui::Context) {
        if self.pacs_server.is_none() {
            return;
        }
        let mut w = self.pacs_server.take().expect("checked above");
        let mut open = true;
        let busy = self.pacs_server_job.is_some();
        let installed = local::installed();
        let mut start = false;
        let mut stop = false;
        let mut new_code = false;
        let mut revoke: Option<String> = None;
        let mut save = false;
        let mut regenerate = false;
        let mut browse_archive = false;
        let mut copy: Option<String> = None;
        let mut allow_firewall = false;

        detach::tool_window(
            ctx,
            "pacs_server",
            "🖥 PACS server",
            &mut open,
            detach::WinOpts::size(700.0, 620.0),
            |ui| {
                ui.label(
                    "Serve this station's archive to other stations over HTTPS, on the local \
                     network or (through a VPN, a forwarded port or a reverse proxy) the \
                     internet. The set-up guide is docs/pacs-server.md.",
                );
                ui.separator();
                if !installed {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!(
                            "⚠ The PACS server is not installed: {} was not found. It is an \
                             optional component of the installer (Install the PACS server), \
                             or cargo build --release --features pacs-server.",
                            settings::pacs_exe_path().display()
                        ),
                    );
                }
                ui.weak(format!(
                    "Configuration: {}",
                    settings::pacs_config_path().display()
                ));

                // ---- state ------------------------------------------------
                ui.horizontal(|ui| match &w.running {
                    Some(r) => {
                        ui.colored_label(egui::Color32::from_rgb(60, 160, 80), "●");
                        ui.label(format!(
                            "running since {} on port {} (version {})",
                            r.started, r.port, r.version
                        ));
                    }
                    None if w.starting => {
                        ui.spinner();
                        ui.label("starting");
                    }
                    None => {
                        ui.weak("● not running");
                    }
                });
                ui.horizontal(|ui| {
                    if enabled_tip_button(
                        ui,
                        installed && !busy && w.running.is_none() && !w.starting,
                        "▶ Start",
                        "Start rds-pacs as a program of its own: it keeps serving when this \
                         window, and the viewer, are closed",
                    ) {
                        start = true;
                    }
                    if enabled_tip_button(
                        ui,
                        !busy && w.running.is_some(),
                        "⏹ Stop",
                        "Stop the server; paired stations keep their pairing",
                    ) {
                        stop = true;
                    }
                });
                if let Some(m) = &w.message {
                    ui.weak(m);
                }
                if let Some(job) = &self.pacs_server_job {
                    let msg = job.progress.get();
                    if !msg.is_empty() && msg != "Asking the server" {
                        ui.weak(msg);
                    }
                }

                // ---- how to reach it -----------------------------------------
                if let Some(r) = &w.running {
                    ui.separator();
                    ui.label(egui::RichText::new("How other stations reach it").strong());
                    // The live list, which notices a link that came or went
                    // since the start; what the server wrote at its start
                    // when there is none (a server bound to one address, or
                    // an interface list that cannot be read).
                    let line_for = |addr: &str| {
                        ConnectionLine::parse(addr).ok().map(|l| ConnectionLine {
                            fingerprint: Some(r.fingerprint.clone()),
                            ..l
                        })
                    };
                    let mut first: Option<String> = None;
                    if w.addresses.is_empty() {
                        for a in &r.addresses {
                            ui.monospace(a);
                        }
                        first = r.addresses.first().cloned();
                    } else {
                        for a in &w.addresses {
                            let with_port = a.with_port(r.port);
                            if first.is_none() {
                                first = Some(with_port.clone());
                            }
                            ui.horizontal(|ui| {
                                ui.monospace(&with_port);
                                if !a.interface.is_empty() {
                                    ui.weak(&a.interface);
                                }
                                match a.link {
                                    Link::Lan => {}
                                    Link::Vpn => {
                                        ui.colored_label(
                                            ui.visuals().warn_fg_color,
                                            format!("({})", a.link.label()),
                                        );
                                    }
                                    Link::Virtual => {
                                        ui.weak(format!("({})", a.link.label()));
                                    }
                                }
                                if let Some(l) = line_for(&with_port) {
                                    if small_tip_button(
                                        ui,
                                        "📋",
                                        "Copy the connection line with this address",
                                    ) {
                                        copy = Some(l.format());
                                    }
                                }
                            });
                        }
                        let host = crate::pacs::this_device_name();
                        if host != "this computer" {
                            ui.horizontal(|ui| {
                                ui.monospace(format!("{host}:{}", r.port));
                                ui.weak("(the name, where the network resolves it)");
                            });
                        }
                    }
                    ui.weak(
                        "A station on the same local network uses the local-network address; \
                         a VPN address works only for stations on that VPN. Over the \
                         internet: a VPN, a forwarded port or a public name (the set-up \
                         guide). An address the router hands out can change; a reservation \
                         in the router keeps it.",
                    );
                    ui.horizontal(|ui| {
                        ui.label("Certificate");
                        ui.label(
                            egui::RichText::new(crate::pacs::show_fingerprint(&r.fingerprint))
                                .monospace(),
                        );
                    });
                    let line = first.as_deref().and_then(line_for);
                    if let Some(line) = &line {
                        if tip_button(
                            ui,
                            "📋 Copy connection details",
                            "The first address and the certificate's fingerprint in one \
                             line, for Tools > PACS > Add server on the other station",
                        ) {
                            copy = Some(line.format());
                        }
                    }

                    // ---- the firewall (Windows) ----------------------------
                    if cfg!(windows) {
                        ui.add_space(4.0);
                        match &w.firewall {
                            Some(Firewall::Allowed(profiles)) => {
                                let on = if profiles.is_empty() {
                                    "every kind of network".to_string()
                                } else {
                                    profiles.join(", ").to_lowercase() + " networks"
                                };
                                ui.weak(format!(
                                    "✔ the Windows firewall lets rds-pacs accept connections on {on}"
                                ));
                            }
                            Some(Firewall::Off) => {
                                ui.weak("The Windows firewall is off: nothing to allow.");
                            }
                            other => {
                                ui.colored_label(
                                    ui.visuals().warn_fg_color,
                                    match other {
                                        Some(Firewall::Blocked) => {
                                            "⚠ A Windows firewall rule blocks rds-pacs (the Windows \
                                             Security Alert was answered with Cancel): no other \
                                             station can reach it."
                                        }
                                        Some(Firewall::NoRule) => {
                                            "⚠ The Windows firewall has no rule for rds-pacs: no \
                                             other station can reach it until one allows it (answer \
                                             Allow when Windows asks, or press the button)."
                                        }
                                        _ => "The Windows firewall's rules could not be read.",
                                    },
                                );
                            }
                        }
                        ui.horizontal(|ui| {
                            if enabled_tip_button(
                                ui,
                                !busy,
                                "🔓 Allow through the Windows firewall",
                                "Make a rule that lets rds-pacs.exe accept TCP connections, \
                                 replacing any rule that blocks it. Windows asks for \
                                 administrator permission.",
                            ) {
                                allow_firewall = true;
                            }
                            ui.checkbox(
                                &mut w.firewall_public,
                                "also on networks Windows calls public",
                            )
                            .on_hover_text(
                                "Windows classes an unknown network, a hotel's or a cafe's, \
                                 as public, and often a home network too until it is marked \
                                 private. The server still answers nothing without a key.",
                            );
                        });
                    }

                    // ---- pairing -------------------------------------------
                    ui.separator();
                    ui.label(egui::RichText::new("Pair a station").strong());
                    ui.horizontal(|ui| {
                        ui.label("Role");
                        egui::ComboBox::from_id_salt("pacs_code_role")
                            .selected_text(w.code_role.label())
                            .show_ui(ui, |ui| {
                                for r in Role::ALL {
                                    ui.selectable_value(&mut w.code_role, r, r.describe());
                                }
                            });
                        ui.label("good for");
                        ui.add(
                            egui::DragValue::new(&mut w.code_minutes)
                                .range(1..=1440)
                                .suffix(" min"),
                        );
                        if enabled_tip_button(
                            ui,
                            !busy,
                            "New pairing code",
                            "A code for one station, used once; tell it to the person \
                             pairing, with the connection details",
                        ) {
                            new_code = true;
                        }
                    });
                    if let Some(c) = &w.code {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(&c.code).monospace().size(22.0).strong());
                            ui.weak(format!(
                                "{} role, good until {}, for one station",
                                c.role.label(),
                                c.expires
                            ));
                        });
                        if let Some(line) = &line {
                            if tip_button(
                                ui,
                                "📋 Copy invitation",
                                "The connection details with this code in them: everything \
                                 the other station needs, in one line. Send it only to the \
                                 person who is to pair.",
                            ) {
                                copy = Some(
                                    ConnectionLine {
                                        code: Some(c.code.clone()),
                                        ..line.clone()
                                    }
                                    .format(),
                                );
                            }
                        }
                    }
                }

                // ---- clients ------------------------------------------------
                ui.separator();
                ui.label(egui::RichText::new("Paired stations").strong());
                if w.clients.is_empty() {
                    ui.weak("none yet");
                }
                egui::Grid::new("pacs_clients")
                    .striped(true)
                    .num_columns(5)
                    .show(ui, |ui| {
                        for c in &w.clients {
                            ui.label(&c.name);
                            ui.label(c.role.label());
                            ui.weak(format!("last seen {}", c.last_seen));
                            ui.weak(&c.address);
                            let asked = w.confirm_revoke.as_deref() == Some(c.name.as_str());
                            if ui
                                .add_enabled(
                                    !busy,
                                    egui::Button::new(if asked {
                                        "Really revoke?"
                                    } else {
                                        "Revoke"
                                    })
                                    .small(),
                                )
                                .on_hover_text("This station's key stops working at once")
                                .clicked()
                            {
                                revoke = Some(c.name.clone());
                            }
                            ui.end_row();
                        }
                    });

                // ---- settings ------------------------------------------------
                ui.separator();
                ui.collapsing("Settings (applied at the next start)", |ui| {
                    let c = &mut w.config;
                    let mut dirty = false;
                    egui::Grid::new("pacs_cfg")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Name");
                            dirty |= ui
                                .add(
                                    egui::TextEdit::singleline(&mut c.name)
                                        .hint_text(crate::pacs::this_device_name())
                                        .desired_width(260.0),
                                )
                                .changed();
                            ui.end_row();
                            ui.label("Port");
                            dirty |= ui
                                .add(egui::DragValue::new(&mut c.port).range(1..=65535))
                                .changed();
                            ui.end_row();
                            ui.label("Listen on");
                            egui::ComboBox::from_id_salt("pacs_bind")
                                .selected_text(match c.bind.as_str() {
                                    "0.0.0.0" => "every network (0.0.0.0)",
                                    "127.0.0.1" => "this computer only (behind a proxy)",
                                    other => other,
                                })
                                .show_ui(ui, |ui| {
                                    for (v, l) in [
                                        ("0.0.0.0", "every network (0.0.0.0)"),
                                        ("127.0.0.1", "this computer only (behind a proxy)"),
                                    ] {
                                        if ui.selectable_label(c.bind == v, l).clicked() {
                                            c.bind = v.into();
                                            c.behind_proxy = v == "127.0.0.1";
                                            dirty = true;
                                        }
                                    }
                                });
                            ui.end_row();
                            ui.label("Archive");
                            ui.horizontal(|ui| {
                                dirty |= ui
                                    .add(
                                        egui::TextEdit::singleline(&mut c.archive_dir)
                                            .hint_text("the station's archive (Tools > PACS)")
                                            .desired_width(300.0),
                                    )
                                    .changed();
                                if ui.button("📂").on_hover_text("Choose a folder").clicked() {
                                    browse_archive = true;
                                }
                            });
                            ui.end_row();
                            ui.label("Extra names");
                            dirty |= ui
                                .add(
                                    egui::TextEdit::singleline(&mut w.advertise)
                                        .hint_text("pacs.example.org, 100.64.0.7")
                                        .desired_width(300.0),
                                )
                                .on_hover_text(
                                    "Names and addresses put into the server's certificate \
                                     (a dynamic DNS name, a VPN address); takes effect with a \
                                     new certificate",
                                )
                                .changed();
                            ui.end_row();
                            ui.label("Uploads up to");
                            dirty |= ui
                                .add(
                                    egui::DragValue::new(&mut c.max_upload_mb)
                                        .range(1..=1_000_000)
                                        .suffix(" MB"),
                                )
                                .changed();
                            ui.end_row();
                        });
                    dirty |= ui
                        .checkbox(&mut c.tasks, "Run the workflows paired stations hand in")
                        .changed();
                    dirty |= ui
                        .checkbox(
                            &mut c.allow_model_download,
                            "Tasks may download model weights they need",
                        )
                        .changed();
                    w.config_dirty |= dirty;
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(w.config_dirty, egui::Button::new("💾 Save"))
                            .on_hover_text(
                                "Write pacs.toml; a running server takes it at its next start",
                            )
                            .clicked()
                        {
                            save = true;
                        }
                        if enabled_tip_button(
                            ui,
                            !busy && w.running.is_none() && w.config.own_certificate().is_none(),
                            "New certificate",
                            "Make a new self-signed certificate (with the extra names). Every \
                             paired station must pair again. Only while the server is stopped.",
                        ) {
                            regenerate = true;
                        }
                    });
                });

                // ---- audit ------------------------------------------------------
                ui.collapsing("Recent activity", |ui| {
                    if w.audit.is_empty() {
                        ui.weak("nothing logged yet");
                    }
                    egui::ScrollArea::vertical()
                        .max_height(160.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for l in &w.audit {
                                ui.monospace(l);
                            }
                        });
                });
            },
        );

        if let Some(text) = copy {
            ctx.copy_text(text);
            w.message = Some("✔ copied to the clipboard".into());
        }
        if let Some(name) = revoke {
            if w.confirm_revoke.as_deref() == Some(name.as_str()) {
                w.confirm_revoke = None;
                let running = w.running.is_some();
                self.server_job("Revoking", move |_| {
                    let paths = Paths::station();
                    if running {
                        let (remote, _) = local::connect(&paths)?;
                        remote.revoke(&name)?;
                    } else {
                        local::revoke_offline(&paths, &name)?;
                    }
                    Ok(ServerOutcome::Revoked(name))
                });
            } else {
                w.confirm_revoke = Some(name);
            }
        }
        if save {
            w.config.advertise = w
                .advertise
                .split([',', ' '])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            match w.config.save(&settings::pacs_config_path()) {
                Ok(()) => {
                    w.config_dirty = false;
                    w.message = Some(if w.running.is_some() {
                        "✔ saved; Stop and Start the server to apply it".into()
                    } else {
                        "✔ saved".into()
                    });
                }
                Err(e) => w.message = Some(format!("⚠ {e:#}")),
            }
        }
        if open {
            self.pacs_server = Some(w);
        }
        let role = self
            .pacs_server
            .as_ref()
            .map(|w| (w.code_role, w.code_minutes));
        if start {
            self.server_job("Starting the server", |_| {
                local::start(&Paths::station())?;
                Ok(ServerOutcome::Started)
            });
        }
        if stop {
            self.server_job("Stopping the server", |_| {
                let (remote, _) = local::connect(&Paths::station())?;
                remote.shutdown()?;
                Ok(ServerOutcome::Stopped)
            });
        }
        if new_code {
            if let Some((role, minutes)) = role {
                self.server_job("Making a pairing code", move |_| {
                    let (remote, _) = local::connect(&Paths::station())?;
                    Ok(ServerOutcome::Code(remote.new_pairing_code(role, minutes)?))
                });
            }
        }
        if regenerate {
            let unsaved = self.pacs_server.as_ref().is_some_and(|w| w.config_dirty);
            if unsaved {
                if let Some(w) = self.pacs_server.as_mut() {
                    w.message = Some(
                        "⚠ save the settings first: the certificate takes the saved names".into(),
                    );
                }
            } else {
                // Only the server's own code makes certificates.
                self.server_job("Making a new certificate", |_| {
                    let out = local::run_cli(&["cert", "--regenerate"])?;
                    Ok(ServerOutcome::Regenerated(
                        crate::pacs::normalize_fingerprint(&out).unwrap_or(out),
                    ))
                });
            }
        }
        if allow_firewall {
            let public = self.pacs_server.as_ref().is_some_and(|w| w.firewall_public);
            self.server_job("Asking Windows for permission", move |_| {
                local::firewall_allow(&Paths::station(), &settings::pacs_exe_path(), public)?;
                Ok(ServerOutcome::FirewallAllowed)
            });
        }
        if browse_archive {
            self.ask_folder("The archive the server serves", |app, dir| {
                if let Some(w) = app.pacs_server.as_mut() {
                    w.config.archive_dir = dir.display().to_string();
                    w.config_dirty = true;
                }
            });
        }
    }
}

/// This machine's addresses for a listener bound to `ip`: both families
/// for `::`, IPv4 only for `0.0.0.0`, and the one address otherwise.
fn live_addresses(ip: IpAddr) -> Vec<Address> {
    if ip.is_unspecified() {
        local::addresses(ip.is_ipv4(), ip.is_ipv6())
    } else {
        Vec::new()
    }
}
