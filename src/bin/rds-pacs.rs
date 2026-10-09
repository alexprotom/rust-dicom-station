//! `rds-pacs` - Rust DICOM Station's archive served to other stations.
//!
//! ```text
//! rds-pacs [--config PATH] [serve]              listen (the default)
//! rds-pacs [--config PATH] --check              print the configuration and exit
//! rds-pacs pair [--role view|edit|run|admin] [--minutes N]
//!                                               make a pairing code (server running)
//! rds-pacs clients [--revoke NAME]              list or revoke paired stations
//! rds-pacs cert [--regenerate]                  show the certificate, or make a new one
//! rds-pacs stop                                 stop the running server
//! ```
//!
//! The configuration is `pacs.toml` in the station's configuration folder
//! (see `docs/pacs-server.md`); the server's state - certificate, clients,
//! the local operator's token - sits beside it in `pacs/`. The viewer's
//! *Settings ▶ PACS server* window starts, watches and stops this same
//! program.
//!
//! Diagnostics go to standard error.

use std::path::PathBuf;

use rust_dicom_station::pacs::{self, config::Config, local, server, Role};

fn usage() -> ! {
    eprintln!(
        "usage: rds-pacs [--config PATH] [serve | --check | pair [--role R] [--minutes N] | \
         clients [--revoke NAME] | cert [--regenerate] | stop]"
    );
    std::process::exit(2);
}

fn fail(e: anyhow::Error) -> ! {
    eprintln!("rds-pacs: {e:#}");
    std::process::exit(1);
}

fn main() {
    let mut config_path: Option<PathBuf> = None;
    let mut command = String::from("serve");
    let mut rest: Vec<String> = Vec::new();
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config_path = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--check" => check = true,
            "-h" | "--help" => usage(),
            "serve" | "pair" | "clients" | "cert" | "stop" if rest.is_empty() => {
                command = a.clone()
            }
            _ => rest.push(a),
        }
    }
    let path = config_path.unwrap_or_else(pacs::config::default_path);
    let cfg = Config::load(&path).unwrap_or_else(|e| fail(e));
    let paths = local::Paths::station();
    if check {
        print_check(&cfg, &paths, &path);
        return;
    }
    match command.as_str() {
        "serve" => serve(cfg, paths),
        "pair" => pair(&paths, &rest),
        "clients" => clients(&paths, &rest),
        "cert" => cert(&cfg, &paths, &rest),
        "stop" => {
            let (remote, _) = local::connect(&paths).unwrap_or_else(|e| fail(e));
            remote.shutdown().unwrap_or_else(|e| fail(e));
            eprintln!("rds-pacs: asked the server to stop");
        }
        _ => usage(),
    }
}

fn print_check(cfg: &Config, paths: &local::Paths, path: &std::path::Path) {
    eprintln!(
        "rds-pacs {}: configuration {}",
        env!("CARGO_PKG_VERSION"),
        if path.is_file() {
            path.display().to_string()
        } else {
            format!("{} (not written: the defaults)", path.display())
        }
    );
    eprintln!("  name     {}", cfg.server_name());
    eprintln!("  listens  {}:{}", cfg.bind, cfg.port);
    eprintln!("  archive  {}", cfg.archive_root().display());
    eprintln!(
        "  tasks    {}{}",
        if cfg.tasks { "on" } else { "off" },
        if cfg.allow_model_download {
            ", model downloads allowed"
        } else {
            ""
        }
    );
    match cfg.own_certificate() {
        Some((c, _)) => eprintln!("  TLS      the operator's certificate {}", c.display()),
        None if paths.cert().is_file() => {
            eprintln!("  TLS      self-signed {}", paths.cert().display())
        }
        None => eprintln!("  TLS      self-signed, made on the first start"),
    }
    match local::Running::read(paths) {
        Some(r) => eprintln!(
            "  running  pid {} on port {} since {}",
            r.pid, r.port, r.started
        ),
        None => eprintln!("  running  no"),
    }
}

fn serve(cfg: Config, paths: local::Paths) {
    if let Some(r) = local::Running::read(&paths) {
        if local::connect(&paths).is_ok() {
            fail(anyhow::anyhow!(
                "a server already runs on this machine (pid {}, port {})",
                r.pid,
                r.port
            ));
        }
    }
    // The engines a task runs read the graphics API from the environment;
    // the viewer's setting is the sensible one to share. Before any thread.
    let preferred = rust_dicom_station::gfx::from_env()
        .unwrap_or_else(|| rust_dicom_station::settings::load().graphics_backend);
    preferred.export();

    let running = server::spawn(cfg, paths.clone()).unwrap_or_else(|e| fail(e));
    let run = local::Running::read(&paths).unwrap_or_default();
    eprintln!(
        "rds-pacs {}: '{}' serving {}",
        env!("CARGO_PKG_VERSION"),
        running.name,
        running.archive.display()
    );
    eprintln!("  listening on {}", running.addr);
    let bind_ip = running.addr.ip();
    let live = if bind_ip.is_unspecified() {
        local::addresses(bind_ip.is_ipv4(), bind_ip.is_ipv6())
    } else {
        Vec::new()
    };
    if live.is_empty() {
        for a in &run.addresses {
            eprintln!("  reachable at {a}");
        }
    } else {
        for a in &live {
            let what = if a.interface.is_empty() {
                String::new()
            } else {
                format!("  {}", a.interface)
            };
            let kind = match a.link {
                local::Link::Lan => String::new(),
                other => format!(" ({})", other.label()),
            };
            eprintln!(
                "  reachable at {}{what}{kind}",
                a.with_port(running.addr.port())
            );
        }
    }
    eprintln!(
        "  certificate  {}",
        pacs::show_fingerprint(&running.fingerprint)
    );
    if let Some(first) = run.addresses.first() {
        if let Ok(line) = pacs::ConnectionLine::parse(first) {
            eprintln!(
                "  connection   {}",
                pacs::ConnectionLine {
                    fingerprint: Some(running.fingerprint.clone()),
                    ..line
                }
                .format()
            );
        }
    }
    // Windows: a listener nobody can reach is the usual first-day problem.
    match local::firewall_state(&std::env::current_exe().unwrap_or_default()) {
        Some(local::Firewall::Allowed(_)) | Some(local::Firewall::Off) | None => {}
        Some(local::Firewall::Blocked) => eprintln!(
            "  note: a Windows firewall rule BLOCKS this program (the Windows Security \
             Alert was declined); other stations cannot reach it. Settings > PACS server > \
             Allow through the Windows firewall in the viewer, or an administrator runs: \
             netsh advfirewall firewall add rule name=\"{}\" dir=in action=allow \
             program=\"{}\" protocol=TCP",
            local::FIREWALL_RULE,
            std::env::current_exe().unwrap_or_default().display()
        ),
        Some(local::Firewall::NoRule) => eprintln!(
            "  note: the Windows firewall has no rule for this program; other stations \
             cannot reach it until one allows it (answer Allow if Windows asks now, or \
             Settings > PACS server > Allow through the Windows firewall in the viewer)"
        ),
    }
    eprintln!("  `rds-pacs pair` makes a pairing code; Ctrl+C stops the server");
    running.stop_on_ctrl_c();
    if let Err(e) = running.wait() {
        fail(e);
    }
}

fn flag(rest: &[String], name: &str) -> Option<String> {
    rest.iter()
        .position(|a| a == name)
        .and_then(|i| rest.get(i + 1).cloned())
}

fn pair(paths: &local::Paths, rest: &[String]) {
    let role = match flag(rest, "--role") {
        Some(r) => Role::from_label(&r).unwrap_or_else(|| usage()),
        None => Role::View,
    };
    let minutes = flag(rest, "--minutes")
        .map(|m| m.parse::<u64>().unwrap_or_else(|_| usage()))
        .unwrap_or(0);
    let (remote, run) = local::connect(paths).unwrap_or_else(|e| fail(e));
    let code = remote
        .new_pairing_code(role, minutes)
        .unwrap_or_else(|e| fail(e));
    println!("{}", code.code);
    eprintln!(
        "pairing code for the {} role, good until {} and for one station",
        code.role.label(),
        code.expires
    );
    if let Some(first) = run.addresses.first() {
        if let Ok(line) = pacs::ConnectionLine::parse(first) {
            eprintln!(
                "invitation: {}",
                pacs::ConnectionLine {
                    fingerprint: Some(run.fingerprint.clone()),
                    code: Some(code.code.clone()),
                    ..line
                }
                .format()
            );
        }
    }
}

fn clients(paths: &local::Paths, rest: &[String]) {
    let revoke = flag(rest, "--revoke");
    match local::connect(paths) {
        Ok((remote, _)) => {
            if let Some(name) = revoke {
                remote.revoke(&name).unwrap_or_else(|e| fail(e));
                eprintln!("revoked '{name}'");
                return;
            }
            print_clients(&remote.clients().unwrap_or_else(|e| fail(e)));
        }
        Err(_) => {
            if let Some(name) = revoke {
                if local::revoke_offline(paths, &name).unwrap_or_else(|e| fail(e)) {
                    eprintln!("revoked '{name}'");
                } else {
                    fail(anyhow::anyhow!("no client '{name}'"));
                }
                return;
            }
            print_clients(&local::clients_offline(paths));
        }
    }
}

fn print_clients(list: &[pacs::protocol::ClientInfo]) {
    if list.is_empty() {
        eprintln!("no station is paired");
    }
    for c in list {
        println!(
            "{:<24} {:<6} paired {}  last seen {}  from {}",
            c.name,
            c.role.label(),
            c.paired,
            c.last_seen,
            c.address
        );
    }
}

fn cert(cfg: &Config, paths: &local::Paths, rest: &[String]) {
    if rest.iter().any(|a| a == "--regenerate") {
        if local::connect(paths).is_ok() {
            fail(anyhow::anyhow!("stop the server first (rds-pacs stop)"));
        }
        if cfg.own_certificate().is_some() {
            fail(anyhow::anyhow!(
                "pacs.toml names a certificate of your own (tls_cert); renew that one instead"
            ));
        }
        let fp = pacs::tls::make(paths, cfg).unwrap_or_else(|e| fail(e));
        println!("{}", pacs::show_fingerprint(&fp));
        eprintln!(
            "a new certificate was made: every paired station must pair again (they will \
             refuse the server until then)"
        );
        return;
    }
    match pacs::tls::load_or_make(paths, cfg) {
        Ok(c) => {
            println!("{}", pacs::show_fingerprint(&c.fingerprint));
            eprintln!(
                "{}",
                if c.self_signed {
                    "self-signed; stations pin this fingerprint when they pair"
                } else {
                    "the operator's certificate from pacs.toml"
                }
            );
        }
        Err(e) => fail(e),
    }
}
