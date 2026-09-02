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
            if models::AVAILABLE {
                ""
            } else {
                "  (not available in this build)"
            }
        );
        println!(
            "Program files       : {}",
            human_size(payload.total_size().unwrap_or(0))
        );
        println!();
        opts.graphics = ask_graphics(opts.graphics)?;
        println!();
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
    if opts.launch_after && !silent && ask_yes_no(&format!("Start {APP_NAME} now?"), false)? {
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
                    "Also delete every downloaded model in {}?",
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

/// The text-mode equivalent of the wizard's graphics page.
///
/// The default is whatever the caller already has - `Vulkan` unless
/// `--graphics` said otherwise - so pressing Enter is always the right
/// answer on a machine with nothing wrong with it.
fn ask_graphics(current: Graphics) -> Result<Graphics> {
    let choices = Graphics::ALL;
    let default_at = choices.iter().position(|g| *g == current).unwrap_or(0);
    println!("Which graphics backend should the viewer start on?");
    for (i, g) in choices.iter().enumerate() {
        println!("  {}) {}", i + 1, g.label());
        for line in wrapped(g.hint(), 68) {
            println!("     {line}");
        }
    }
    loop {
        print!("Choice [{}] ", default_at + 1);
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            return Ok(current);
        }
        let line = line.trim();
        if line.is_empty() {
            return Ok(current);
        }
        // A number from the list, or the name itself - both are things
        // people type, and neither is worth rejecting.
        if let Ok(n) = line.parse::<usize>() {
            if (1..=choices.len()).contains(&n) {
                return Ok(choices[n - 1]);
            }
        }
        if let Some(g) = Graphics::from_key(line) {
            return Ok(g);
        }
        println!("Please answer with a number from 1 to {}.", choices.len());
    }
}

/// Break a sentence into lines no longer than `width`, so a hint written as
/// one long string does not arrive as one long console line.
fn wrapped(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if !cur.is_empty() && cur.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
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
