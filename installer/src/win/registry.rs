//! A minimal registry wrapper: create/open keys, read/write string and DWORD
//! values, delete whole subtrees, and edit the `PATH` environment value.

use anyhow::{Context, Result};
use windows::core::PCWSTR;
use windows::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteTreeW, RegDeleteValueW, RegOpenKeyExW,
    RegQueryValueExW, RegSetValueExW, HKEY, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ,
    KEY_WRITE, REG_DWORD, REG_EXPAND_SZ, REG_OPTION_NON_VOLATILE, REG_SZ, REG_VALUE_TYPE,
};

use super::wide;

/// Which hive an installation writes to: per-user or machine-wide.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hive {
    CurrentUser,
    LocalMachine,
}

impl Hive {
    pub fn hkey(self) -> HKEY {
        match self {
            Hive::CurrentUser => HKEY_CURRENT_USER,
            Hive::LocalMachine => HKEY_LOCAL_MACHINE,
        }
    }

    /// Registry path of the environment block this hive owns.
    pub fn environment_key(self) -> &'static str {
        match self {
            Hive::CurrentUser => "Environment",
            Hive::LocalMachine => {
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Hive::CurrentUser => "HKEY_CURRENT_USER",
            Hive::LocalMachine => "HKEY_LOCAL_MACHINE",
        }
    }
}

/// An owned, automatically closed registry key handle.
pub struct Key(HKEY);

impl Drop for Key {
    fn drop(&mut self) {
        unsafe {
            let _ = RegCloseKey(self.0);
        }
    }
}

impl Key {
    /// Create `root\path` (including missing parents), or open it if it exists.
    pub fn create(root: HKEY, path: &str) -> Result<Key> {
        let p = wide(path);
        let mut key = HKEY::default();
        unsafe {
            RegCreateKeyExW(
                root,
                PCWSTR(p.as_ptr()),
                None,
                PCWSTR::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_READ | KEY_WRITE,
                None,
                &mut key,
                None,
            )
            .ok()
            .with_context(|| format!("create registry key {path}"))?;
        }
        Ok(Key(key))
    }

    /// Open an existing key for reading (and writing when `write`).
    pub fn open(root: HKEY, path: &str, write: bool) -> Result<Key> {
        let p = wide(path);
        let access = if write { KEY_READ | KEY_WRITE } else { KEY_READ };
        let mut key = HKEY::default();
        unsafe {
            RegOpenKeyExW(root, PCWSTR(p.as_ptr()), None, access, &mut key)
                .ok()
                .with_context(|| format!("open registry key {path}"))?;
        }
        Ok(Key(key))
    }

    fn set_raw(&self, name: &str, ty: REG_VALUE_TYPE, data: &[u8]) -> Result<()> {
        let n = wide(name);
        unsafe {
            RegSetValueExW(self.0, PCWSTR(n.as_ptr()), None, ty, Some(data))
                .ok()
                .with_context(|| format!("write registry value {name}"))?;
        }
        Ok(())
    }

    /// Write a `REG_SZ` value. An empty `name` writes the key's default value.
    pub fn set_str(&self, name: &str, value: &str) -> Result<()> {
        self.set_raw(name, REG_SZ, as_bytes(&wide(value)))
    }

    /// Write a `REG_EXPAND_SZ` value (one that may contain `%VARIABLES%`).
    pub fn set_expand_str(&self, name: &str, value: &str) -> Result<()> {
        self.set_raw(name, REG_EXPAND_SZ, as_bytes(&wide(value)))
    }

    pub fn set_u32(&self, name: &str, value: u32) -> Result<()> {
        self.set_raw(name, REG_DWORD, &value.to_le_bytes())
    }

    /// Read a string value, together with its type so a `PATH` containing
    /// `%VARIABLES%` can be written back unexpanded.
    pub fn get_str_typed(&self, name: &str) -> Option<(String, REG_VALUE_TYPE)> {
        let n = wide(name);
        let mut ty = REG_VALUE_TYPE::default();
        let mut len: u32 = 0;
        unsafe {
            RegQueryValueExW(
                self.0,
                PCWSTR(n.as_ptr()),
                None,
                Some(&mut ty),
                None,
                Some(&mut len),
            )
            .ok()
            .ok()?;
            let mut buf = vec![0u8; len as usize];
            RegQueryValueExW(
                self.0,
                PCWSTR(n.as_ptr()),
                None,
                Some(&mut ty),
                Some(buf.as_mut_ptr()),
                Some(&mut len),
            )
            .ok()
            .ok()?;
            let units: Vec<u16> = buf
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .take_while(|&u| u != 0)
                .collect();
            Some((String::from_utf16_lossy(&units), ty))
        }
    }

    pub fn get_str(&self, name: &str) -> Option<String> {
        self.get_str_typed(name).map(|(s, _)| s)
    }

    pub fn delete_value(&self, name: &str) -> Result<()> {
        let n = wide(name);
        unsafe {
            RegDeleteValueW(self.0, PCWSTR(n.as_ptr()))
                .ok()
                .with_context(|| format!("delete registry value {name}"))?;
        }
        Ok(())
    }
}

fn as_bytes(units: &[u16]) -> &[u8] {
    // SAFETY: reinterpreting UTF-16 units as the little-endian byte buffer the
    // registry expects; length is exact and alignment only decreases.
    unsafe { std::slice::from_raw_parts(units.as_ptr() as *const u8, units.len() * 2) }
}

/// Delete `root\path` and everything below it. Missing keys are not an error.
pub fn delete_tree(root: HKEY, path: &str) -> Result<()> {
    let p = wide(path);
    unsafe {
        let err = RegDeleteTreeW(root, PCWSTR(p.as_ptr()));
        if err.is_ok() {
            return Ok(());
        }
        // ERROR_FILE_NOT_FOUND (2) — already gone.
        if err.0 == 2 {
            return Ok(());
        }
        Err(anyhow::anyhow!("delete registry key {path}: error {}", err.0))
    }
}

/// Append `dir` to the hive's `PATH` if it is not already listed.
/// Returns true when the value was actually changed.
pub fn path_add(hive: Hive, dir: &str) -> Result<bool> {
    let key = Key::create(hive.hkey(), hive.environment_key())?;
    let (current, ty) = key
        .get_str_typed("Path")
        .unwrap_or_else(|| (String::new(), REG_EXPAND_SZ));
    if path_contains(&current, dir) {
        return Ok(false);
    }
    let mut new = current.trim_end_matches(';').to_string();
    if !new.is_empty() {
        new.push(';');
    }
    new.push_str(dir);
    if ty == REG_SZ && !new.contains('%') {
        key.set_str("Path", &new)?;
    } else {
        key.set_expand_str("Path", &new)?;
    }
    Ok(true)
}

/// Remove `dir` from the hive's `PATH`. Returns true when something changed.
pub fn path_remove(hive: Hive, dir: &str) -> Result<bool> {
    let Ok(key) = Key::open(hive.hkey(), hive.environment_key(), true) else {
        return Ok(false);
    };
    let Some((current, ty)) = key.get_str_typed("Path") else {
        return Ok(false);
    };
    if !path_contains(&current, dir) {
        return Ok(false);
    }
    let kept: Vec<&str> = current
        .split(';')
        .filter(|p| !same_path(p, dir))
        .filter(|p| !p.trim().is_empty())
        .collect();
    let new = kept.join(";");
    if ty == REG_SZ && !new.contains('%') {
        key.set_str("Path", &new)?;
    } else {
        key.set_expand_str("Path", &new)?;
    }
    Ok(true)
}

fn path_contains(list: &str, dir: &str) -> bool {
    list.split(';').any(|p| same_path(p, dir))
}

fn same_path(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        s.trim()
            .trim_matches('"')
            .trim_end_matches(['\\', '/'])
            .to_ascii_lowercase()
            .replace('/', "\\")
    };
    !a.trim().is_empty() && norm(a) == norm(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_membership_ignores_case_slashes_and_trailing_separators() {
        let list = r"C:\Windows;C:\Program Files\Rust DICOM Viewer\;D:\bin";
        assert!(path_contains(list, r"c:\program files\rust dicom viewer"));
        assert!(path_contains(list, r"C:/Program Files/Rust DICOM Viewer"));
        assert!(!path_contains(list, r"C:\Program Files\Other"));
        assert!(!path_contains(";;", r""), "empty entries never match");
    }
}
