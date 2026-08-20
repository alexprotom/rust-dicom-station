//! Text-mode front end: used for `--silent`, for `--console`, and as the
//! automatic fallback when no window can be created (no usable GPU adapter,
//! a session without a desktop, …).

use std::io::Write;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use anyhow::Result;

use crate::install::{self, Event};
use crate::models;
use crate::payload::Payload;
use crate::plan::*;
use crate::uninstall::{self, Target};

pub fn run_install(payload: Payload, mut opts: Options, silent: bool) -> Result<()> {
    println!("\n{APP_NAME} setup\n");
    if !silent {
        println!("Install folder      : {}", opts.dir.display());
        println!("Scope               : {}", opts.scope.label());
        println!("Start menu shortcut : {}", yes_no(opts.start_menu_shortcut));
        println!("Desktop shortcut    : {}", yes_no(opts.desktop_shortcut));
        println!("File association    : {}", yes_no(opts.file_association));
        println!("Add to PATH         : {}", yes_no(opts.add_to_path));
        println!(
            "Model weights       : {}{}",
            opts.models.label(),
            if models::AVAILABLE { "" } else { "  (not available in this build)" }
        );
        println!(
            "Program files       : {}\n",
            human_size(payload.total_size().unwrap_or(0))
        );
        if !ask_yes_no("Proceed with the installation?", true)? {
            println!("Cancelled.");
            return Ok(());
        }
    }
    if !models::AVAILABLE {
        opts.models = Models::None;
    }
    let cancel = AtomicBool::new(false);
    let printer = Printer::default();
    let sink = |ev: Event| printer.handle(ev);
    install::run(&opts, &payload, &sink, &cancel)?;
    println!("\nInstalled into {}", opts.dir.display());
    if opts.launch_after && !silent && ask_yes_no("Start the viewer now?", false)? {
        crate::win::shell_execute(&opts.exe_path(), "", false)?;
    }
    Ok(())
}

pub fn run_uninstall(target: Target, mut remove_models: bool, silent: bool) -> Result<()> {
    println!("\nUninstall {APP_NAME}\n");
    println!("Install folder : {}", target.manifest.install_dir.display());
    if !silent {
        if !ask_yes_no("Remove this installation?", true)? {
            println!("Cancelled.");
            return Ok(());
        }
        if !remove_models && target.manifest.models_dir.exists() {
            remove_models = ask_yes_no(
                &format!(
                    "Also delete the model weights in {}?",
                    target.manifest.models_dir.display()
                ),
                false,
            )?;
        }
    }
    let printer = Printer::default();
    let sink = |ev: Event| printer.handle(ev);
    uninstall::run(&target, remove_models, &sink)?;
    println!("\nDone.");
    Ok(())
}

/// Prints log lines immediately and progress at most once per whole percent.
#[derive(Default)]
struct Printer {
    last: Mutex<(i32, String)>,
}

impl Printer {
    fn handle(&self, ev: Event) {
        match ev {
            Event::Log(line) => println!("  {line}"),
            Event::Progress(frac, msg) => {
                let pct = (frac * 100.0).round() as i32;
                let mut last = self.last.lock().unwrap();
                if pct != last.0 || msg != last.1 {
                    *last = (pct, msg.clone());
                    print!("\r[{pct:3}%] {msg:<60}");
                    let _ = std::io::stdout().flush();
                    if frac >= 1.0 {
                        println!();
                    }
                }
            }
        }
    }
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

fn ask_yes_no(question: &str, default: bool) -> Result<bool> {
    let hint = if default { "[Y/n]" } else { "[y/N]" };
    loop {
        print!("{question} {hint} ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            return Ok(default);
        }
        match line.trim().to_ascii_lowercase().as_str() {
            "" => return Ok(default),
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("Please answer y or n."),
        }
    }
}
