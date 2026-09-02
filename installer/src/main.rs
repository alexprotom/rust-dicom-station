//! `rds-setup` — the Windows installer for rust-dicom-station.
//!
//! Run without arguments it shows a small wizard; `--silent` and `--console`
//! drive the same code from a terminal, and `--uninstall` (the form recorded
//! in Apps & features) removes an installation again.
//!
//! The binary is built as a GUI-subsystem program so double-clicking it does
//! not flash a console window; the text interface attaches to the parent
//! console explicitly.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(not(windows))]
compile_error!("rds-setup installs a Windows application and only builds for Windows targets");

mod console;
mod deps;
mod install;
mod models;
mod payload;
mod plan;
mod ui;
mod uninstall;
mod win;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Result};

use payload::Payload;
use plan::*;

const USAGE: &str = "\
Rust DICOM Station setup

USAGE:
    rds-setup [OPTIONS]
    rds-setup --uninstall [--remove-models] [--silent]

INSTALL OPTIONS:
    --dir <PATH>          destination folder
    --all-users           install for all users (needs administrator rights)
    --just-me             install for the current user only (default)
    --models <SET>        pre-download weights: none | 6mm | 3mm | 1.5mm | all
    --models-dir <PATH>   the model folder (every engine's weights go in it)
    --no-start-menu       skip the Start-menu shortcut
    --no-desktop-shortcut skip the desktop shortcut
    --no-file-association skip the .dcm / folder context-menu entries
    --add-to-path         add the program folder to PATH
    --no-vcredist         do not install the Visual C++ runtime when missing
    --no-launch           do not offer to start the viewer afterwards
    --graphics <API>      which graphics API the viewer starts on:
                          vulkan (default) | dx12 | auto

UNINSTALL OPTIONS:
    --uninstall           remove an existing installation
    --from <PATH>         installation folder (default: the uninstaller's own)
    --remove-models       also delete the model folder with every downloaded model

GENERAL:
    --silent              no window and no questions
    --console             text interface instead of the wizard
    -h, --help            show this help
";

#[derive(Default)]
struct Args {
    help: bool,
    silent: bool,
    console: bool,
    elevated: bool,
    uninstall: bool,
    remove_models: bool,
    from: Option<PathBuf>,
    opts: Option<Options>,
}

fn parse_args() -> Result<Args> {
    parse_from(std::env::args().skip(1))
}

/// The command line, taken from an iterator so the round trip through
/// [`args_for_relaunch`] can be tested: everything the user chose in the
/// first window has to survive the elevated re-launch, and a flag quietly
/// dropped there loses a choice without any sign of it.
fn parse_from(args: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args::default();
    let mut opts = Options::default();
    let mut it = args;
    let next = |it: &mut dyn Iterator<Item = String>, flag: &str| -> Result<String> {
        it.next()
            .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" | "/?" => a.help = true,
            "--silent" | "/S" | "/silent" => a.silent = true,
            "--console" => a.console = true,
            "--elevated" => a.elevated = true,
            "--uninstall" | "/uninstall" => a.uninstall = true,
            "--remove-models" => a.remove_models = true,
            "--from" => a.from = Some(PathBuf::from(next(&mut it, "--from")?)),
            "--dir" => opts.set_dir(PathBuf::from(next(&mut it, "--dir")?)),
            "--models-dir" => opts.models_dir = PathBuf::from(next(&mut it, "--models-dir")?),
            "--all-users" => opts.set_scope(Scope::AllUsers),
            "--just-me" => opts.set_scope(Scope::CurrentUser),
            "--no-start-menu" => opts.start_menu_shortcut = false,
            "--no-desktop-shortcut" => opts.desktop_shortcut = false,
            "--no-file-association" => opts.file_association = false,
            "--add-to-path" => opts.add_to_path = true,
            "--no-vcredist" => opts.install_vcredist = false,
            "--no-launch" => opts.launch_after = false,
            "--graphics" => {
                let v = next(&mut it, "--graphics")?;
                opts.graphics = Graphics::from_key(&v)
                    .ok_or_else(|| anyhow::anyhow!("unknown --graphics value '{v}'"))?;
            }
            "--models" => {
                let v = next(&mut it, "--models")?;
                opts.models = match v.to_ascii_lowercase().as_str() {
                    "none" => Models::None,
                    "6mm" => Models::Preview6mm,
                    "3mm" | "fast" => Models::Fast3mm,
                    "1.5mm" | "15mm" | "highres" => Models::HighRes15mm,
                    "all" | "everything" => Models::Everything,
                    other => bail!("unknown --models value '{other}'"),
                };
            }
            other => bail!("unknown option '{other}'\n\n{USAGE}"),
        }
    }
    a.opts = Some(opts);
    Ok(a)
}

/// Re-serialise the chosen options for the elevated re-launch.
pub fn args_for_relaunch(o: &Options) -> String {
    let mut s = format!(
        "--elevated --dir \"{}\" --models-dir \"{}\"",
        o.dir.display(),
        o.models_dir.display()
    );
    s.push_str(match o.scope {
        Scope::AllUsers => " --all-users",
        Scope::CurrentUser => " --just-me",
    });
    if !o.start_menu_shortcut {
        s.push_str(" --no-start-menu");
    }
    if !o.desktop_shortcut {
        s.push_str(" --no-desktop-shortcut");
    }
    if !o.file_association {
        s.push_str(" --no-file-association");
    }
    if o.add_to_path {
        s.push_str(" --add-to-path");
    }
    if !o.install_vcredist {
        s.push_str(" --no-vcredist");
    }
    let models = match o.models {
        Models::None => "none",
        Models::Preview6mm => "6mm",
        Models::Fast3mm => "3mm",
        Models::HighRes15mm => "1.5mm",
        Models::Everything => "all",
    };
    s.push_str(&format!(" --models {models}"));
    // The elevated run writes the settings file, so it has to be told which
    // backend the user chose on the graphics page.
    s.push_str(&format!(" --graphics {}", o.graphics.key()));
    // The elevated process must not start the viewer: it would inherit the
    // administrator token. The first (unelevated) window is gone by then, so
    // the user starts it from the shortcut instead.
    s.push_str(" --no-launch");
    s
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            win::attach_console();
            eprintln!("{e:#}");
            return ExitCode::FAILURE;
        }
    };
    if args.help {
        win::attach_console();
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let text_mode = args.silent || args.console;
    if text_mode {
        win::attach_console();
    }
    let result = if args.uninstall {
        do_uninstall(&args)
    } else {
        do_install(&args)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let msg = format!("{e:#}");
            if text_mode {
                eprintln!("\nError: {msg}");
            } else {
                win::message_box(&format!("{APP_NAME} setup"), &msg);
            }
            ExitCode::FAILURE
        }
    }
}

fn do_install(args: &Args) -> Result<()> {
    let payload = Payload::locate()?;
    let mut opts = args.opts.clone().unwrap_or_default();
    if !models::AVAILABLE {
        opts.models = Models::None;
    }
    // A machine-wide installation needs elevation. The wizard asks for it when
    // the user presses Install; the silent and console paths have no way to ask,
    // so they fail here instead.
    if opts.scope == Scope::AllUsers && !win::is_elevated() && (args.silent || args.console) {
        bail!(
            "a machine-wide installation needs administrator rights - start the installer \
             from an elevated prompt, or install with --just-me"
        );
    }
    if args.silent || args.console {
        return console::run_install(payload, opts, args.silent);
    }
    match ui::run_install(payload, opts.clone(), args.elevated) {
        Ok(()) => Ok(()),
        Err(gui_err) => {
            // No window? Carry on in the terminal rather than failing outright.
            win::attach_console();
            eprintln!("{gui_err:#}\nFalling back to the text interface.\n");
            let payload = Payload::locate()?;
            console::run_install(payload, opts, false)
        }
    }
}

fn do_uninstall(args: &Args) -> Result<()> {
    let target = uninstall::discover(args.from.clone())?;
    // A machine-wide installation lives in Program Files and owns HKLM keys;
    // Apps & features starts the uninstaller unelevated, so ask for rights.
    if target.manifest.machine_wide && !win::is_elevated() {
        let mut relaunch = format!("--uninstall --from \"{}\"", target.dir.display());
        if args.silent {
            relaunch.push_str(" --silent");
        }
        if args.remove_models {
            relaunch.push_str(" --remove-models");
        }
        win::relaunch_elevated(&relaunch)?;
        return Ok(());
    }
    // uninstall.exe lives in the folder it has to delete: continue from a copy
    // in %TEMP%, which can remove the original.
    if uninstall::running_from_target(&target) {
        let mut extra: Vec<&str> = Vec::new();
        if args.silent {
            extra.push("--silent");
        }
        if args.console {
            extra.push("--console");
        }
        if args.remove_models {
            extra.push("--remove-models");
        }
        uninstall::relaunch_from_temp(&target, &extra)?;
        return Ok(());
    }
    // Running from %TEMP%: clean the copy up at the next boot.
    if let Ok(exe) = std::env::current_exe() {
        if exe.starts_with(std::env::temp_dir()) {
            win::delete_on_reboot(&exe);
        }
    }
    if args.silent || args.console {
        return console::run_uninstall(target, args.remove_models, args.silent);
    }
    match ui::run_uninstall(target) {
        Ok(()) => Ok(()),
        Err(gui_err) => {
            win::attach_console();
            eprintln!("{gui_err:#}\nFalling back to the text interface.\n");
            let target = uninstall::discover(args.from.clone())?;
            console::run_uninstall(target, args.remove_models, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Args {
        // Good enough for the strings `args_for_relaunch` produces: the only
        // quoted values are the two paths.
        let mut words: Vec<String> = Vec::new();
        let mut cur = String::new();
        let mut quoted = false;
        for c in line.chars() {
            match c {
                '"' => quoted = !quoted,
                ' ' if !quoted => {
                    if !cur.is_empty() {
                        words.push(std::mem::take(&mut cur));
                    }
                }
                _ => cur.push(c),
            }
        }
        if !cur.is_empty() {
            words.push(cur);
        }
        parse_from(words.into_iter()).expect("the installer must be able to read its own output")
    }

    /// The user answers every page in the first window; a machine-wide
    /// installation then throws that window away and starts again as
    /// administrator with nothing but this command line. Anything missing
    /// from it is a choice silently lost.
    #[test]
    fn every_choice_survives_the_elevated_relaunch() {
        for graphics in Graphics::ALL {
            for models in Models::ALL {
                let before = Options {
                    dir: PathBuf::from(r"D:\Apps\Rust DICOM Station"),
                    models_dir: PathBuf::from(r"D:\weights"),
                    scope: Scope::AllUsers,
                    start_menu_shortcut: false,
                    desktop_shortcut: false,
                    file_association: false,
                    add_to_path: true,
                    install_vcredist: false,
                    launch_after: true,
                    models,
                    graphics,
                };
                let after = parse(&args_for_relaunch(&before))
                    .opts
                    .expect("parse_from always fills in the options");
                assert_eq!(after.dir, before.dir);
                assert_eq!(after.models_dir, before.models_dir);
                assert_eq!(after.scope, before.scope);
                assert_eq!(after.start_menu_shortcut, before.start_menu_shortcut);
                assert_eq!(after.desktop_shortcut, before.desktop_shortcut);
                assert_eq!(after.file_association, before.file_association);
                assert_eq!(after.add_to_path, before.add_to_path);
                assert_eq!(after.install_vcredist, before.install_vcredist);
                assert_eq!(after.models, before.models);
                assert_eq!(after.graphics, before.graphics, "the graphics page");
                // The one deliberate difference: the elevated process must
                // not start the viewer, or it would inherit the token.
                assert!(!after.launch_after);
            }
        }
    }

    #[test]
    fn the_graphics_flag_takes_the_spellings_people_type() {
        for (text, want) in [
            ("vulkan", Graphics::Vulkan),
            ("DX12", Graphics::Dx12),
            ("directx", Graphics::Dx12),
            ("auto", Graphics::Auto),
        ] {
            let a = parse_from(["--graphics".to_string(), text.to_string()].into_iter()).unwrap();
            assert_eq!(a.opts.unwrap().graphics, want, "--graphics {text}");
        }
        assert!(
            parse_from(["--graphics".to_string(), "opengl".to_string()].into_iter()).is_err(),
            "a backend the installer does not offer is refused rather than ignored"
        );
        assert!(
            parse_from(["--graphics".to_string()].into_iter()).is_err(),
            "and the flag needs a value"
        );
    }

    /// Without `--graphics` the installer must still pick Vulkan, which is
    /// what the wizard shows preselected.
    #[test]
    fn vulkan_is_the_default() {
        let a = parse_from(std::iter::empty()).unwrap();
        assert_eq!(a.opts.unwrap().graphics, Graphics::Vulkan);
    }
}
