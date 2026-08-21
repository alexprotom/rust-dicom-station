//! Thin, hand-rolled wrappers over the Win32 APIs the installer needs.
//!
//! Everything here is a direct call into the operating system through the
//! `windows` crate — no third-party installer framework, in keeping with the
//! project's one-language rule.

pub mod registry;
pub mod shortcut;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use windows::core::{GUID, HSTRING, PCWSTR, PWSTR};
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::Storage::FileSystem::{
    CreateFileW, MoveFileExW, FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_SHARE_READ,
    FILE_SHARE_WRITE, MOVEFILE_DELAY_UNTIL_REBOOT, OPEN_EXISTING,
};
use windows::Win32::System::Com::CoTaskMemFree;
use windows::Win32::System::Console::{
    AllocConsole, AttachConsole, GetStdHandle, SetStdHandle, ATTACH_PARENT_PROCESS,
    STD_ERROR_HANDLE, STD_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
};
use windows::Win32::UI::Shell::{
    FOLDERID_Desktop, FOLDERID_LocalAppData, FOLDERID_ProgramFiles, FOLDERID_Programs,
    IsUserAnAdmin, SHGetKnownFolderPath, ShellExecuteW, KF_FLAG_DEFAULT,
};
use windows::Win32::UI::WindowsAndMessaging::{
    MessageBoxW, SendMessageTimeoutW, HWND_BROADCAST, MB_ICONERROR, MB_OK, SMTO_ABORTIFHUNG,
    SW_SHOWNORMAL, WM_SETTINGCHANGE,
};

/// NUL-terminated UTF-16 for the Win32 `W` entry points.
pub fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn known_folder(id: &GUID) -> Result<PathBuf> {
    unsafe {
        let p: PWSTR =
            SHGetKnownFolderPath(id, KF_FLAG_DEFAULT, None).context("SHGetKnownFolderPath")?;
        let s = p
            .to_string()
            .context("known folder path is not valid UTF-16")?;
        CoTaskMemFree(Some(p.0 as *const _));
        Ok(PathBuf::from(s))
    }
}

/// `%LOCALAPPDATA%` — resolved through the shell, so folder redirection and
/// roaming profiles are honoured.
pub fn local_app_data() -> Result<PathBuf> {
    known_folder(&FOLDERID_LocalAppData)
}

/// `C:\Program Files`.
pub fn program_files() -> Result<PathBuf> {
    known_folder(&FOLDERID_ProgramFiles)
}

/// The user's Desktop — the real one, which may live under OneDrive.
pub fn desktop_dir() -> Result<PathBuf> {
    known_folder(&FOLDERID_Desktop)
}

/// The user's Start Menu ▸ Programs folder.
pub fn start_menu_programs() -> Result<PathBuf> {
    known_folder(&FOLDERID_Programs)
}

/// True when the process runs with an elevated (administrator) token.
pub fn is_elevated() -> bool {
    unsafe { IsUserAnAdmin().as_bool() }
}

/// Re-launch ourselves elevated with `args`, returning `Ok(())` once the UAC
/// prompt has been accepted and the new process started.
pub fn relaunch_elevated(args: &str) -> Result<()> {
    let exe = std::env::current_exe()?;
    let exe = HSTRING::from(exe.as_os_str());
    let args = HSTRING::from(args);
    let verb = HSTRING::from("runas");
    let inst = unsafe {
        ShellExecuteW(
            None,
            PCWSTR(verb.as_ptr()),
            PCWSTR(exe.as_ptr()),
            PCWSTR(args.as_ptr()),
            PCWSTR::null(),
            SW_SHOWNORMAL,
        )
    };
    // ShellExecuteW reports failure as a "HINSTANCE" <= 32.
    if inst.0 as usize <= 32 {
        anyhow::bail!("the elevation request was declined or failed");
    }
    Ok(())
}

/// Start a program detached from the installer (used for "launch the viewer
/// now" and for running the Visual C++ redistributable).
pub fn shell_execute(file: &Path, args: &str, wait: bool) -> Result<()> {
    let file_h = HSTRING::from(file.as_os_str());
    let args_h = HSTRING::from(args);
    let dir_h = file
        .parent()
        .map(|p| HSTRING::from(p.as_os_str()))
        .unwrap_or_default();
    if wait {
        // Waiting needs a real child handle, which std::process gives us.
        let status = std::process::Command::new(file)
            .args(args.split_whitespace())
            .status()
            .with_context(|| format!("run {}", file.display()))?;
        if !status.success() {
            anyhow::bail!("{} exited with {}", file.display(), status);
        }
        return Ok(());
    }
    let inst = unsafe {
        ShellExecuteW(
            None,
            PCWSTR::null(),
            PCWSTR(file_h.as_ptr()),
            PCWSTR(args_h.as_ptr()),
            PCWSTR(dir_h.as_ptr()),
            SW_SHOWNORMAL,
        )
    };
    if inst.0 as usize <= 32 {
        anyhow::bail!("could not start {}", file.display());
    }
    Ok(())
}

/// Tell every running program that the environment block changed, so freshly
/// opened shells see an updated `PATH` without a logoff.
pub fn broadcast_environment_change() {
    let env = wide("Environment");
    unsafe {
        SendMessageTimeoutW(
            HWND_BROADCAST,
            WM_SETTINGCHANGE,
            WPARAM(0),
            LPARAM(env.as_ptr() as isize),
            SMTO_ABORTIFHUNG,
            5_000,
            None,
        );
    }
}

/// Schedule `path` for deletion at the next boot. Used by the uninstaller for
/// the temporary copy of itself, which cannot delete its own running image.
pub fn delete_on_reboot(path: &Path) {
    let h = HSTRING::from(path.as_os_str());
    unsafe {
        let _ = MoveFileExW(
            PCWSTR(h.as_ptr()),
            PCWSTR::null(),
            MOVEFILE_DELAY_UNTIL_REBOOT,
        );
    }
}

/// Give this GUI-subsystem process a console: attach to the parent one when
/// started from a shell, otherwise allocate a fresh window. Standard handles
/// are then rebound to `CONOUT$`/`CONIN$` so `println!` and `read_line` work
/// — but only the ones the process does not already have, so redirection
/// (`rds-setup --silent > log.txt`) keeps working.
pub fn attach_console() {
    unsafe {
        if AttachConsole(ATTACH_PARENT_PROCESS).is_err() && AllocConsole().is_err() {
            return;
        }
        bind_std_handle(STD_OUTPUT_HANDLE, "CONOUT$");
        bind_std_handle(STD_ERROR_HANDLE, "CONOUT$");
        bind_std_handle(STD_INPUT_HANDLE, "CONIN$");
    }
}

unsafe fn bind_std_handle(which: STD_HANDLE, device: &str) {
    if let Ok(existing) = GetStdHandle(which) {
        if !existing.is_invalid() {
            return; // inherited from the parent (a pipe or a file): leave it.
        }
    }
    let h = CreateFileW(
        PCWSTR(wide(device).as_ptr()),
        (FILE_GENERIC_READ | FILE_GENERIC_WRITE).0,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        Default::default(),
        None,
    );
    if let Ok(h) = h {
        let _ = SetStdHandle(which, h);
    }
}

/// A plain message box — the only way to report a fatal error when the
/// installer runs windowed and the wizard never came up.
pub fn message_box(title: &str, text: &str) {
    let t = HSTRING::from(title);
    let b = HSTRING::from(text);
    unsafe {
        MessageBoxW(
            None,
            PCWSTR(b.as_ptr()),
            PCWSTR(t.as_ptr()),
            MB_ICONERROR | MB_OK,
        );
    }
}
