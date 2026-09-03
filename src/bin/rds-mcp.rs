//! `rds-mcp` - Rust DICOM Station as a Model Context Protocol server.
//!
//! Usage: `rds-mcp [--config PATH] [--check]`
//!
//! Speaks MCP over standard input and output, which is how MCP clients
//! (Claude Desktop, Claude Code and others) launch a server themselves. The
//! configuration - which folders may be read, where results go, what happens
//! to a dataset that still names its patient - comes from `mcp.toml` in the
//! station's configuration folder and never from the client. `--check` reads
//! the configuration, prints what it says on standard error, and exits.
//!
//! Diagnostics go to standard error; standard output belongs to the protocol.

use std::path::PathBuf;

use rust_dicom_station::mcp::{config, Config, Core};

fn usage() -> ! {
    eprintln!("usage: rds-mcp [--config PATH] [--check]");
    std::process::exit(2);
}

fn main() {
    let mut config_path: Option<PathBuf> = None;
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => config_path = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            "--check" => check = true,
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    let path = config_path.unwrap_or_else(config::default_path);
    let cfg = match Config::load(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("rds-mcp: {e:#}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "rds-mcp {}: {} root(s), output {}, PHI policy '{}', device {}, downloads {}",
        env!("CARGO_PKG_VERSION"),
        cfg.roots.len(),
        if cfg.output_dir.is_some() {
            "configured"
        } else {
            "not configured"
        },
        cfg.phi_policy.label(),
        cfg.device,
        if cfg.allow_model_download {
            "allowed"
        } else {
            "off"
        },
    );
    if cfg.roots.is_empty() {
        eprintln!(
            "rds-mcp: no roots in {}; no dataset can be opened until some are configured",
            path.display()
        );
    }
    if check {
        return;
    }
    // The inference backend reads the graphics API from the environment; the
    // viewer's setting is the sensible one to share.
    let preferred = rust_dicom_station::gfx::from_env()
        .unwrap_or_else(|| rust_dicom_station::settings::load().graphics_backend);
    preferred.export();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("a tokio runtime");
    let result = rt.block_on(rust_dicom_station::mcp::server::serve_stdio(Core::new(cfg)));
    if let Err(e) = result {
        eprintln!("rds-mcp: {e:#}");
        std::process::exit(1);
    }
}
