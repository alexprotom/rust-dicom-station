//! The installer payload — the files that end up in the install directory.
//!
//! A shipped `rust-dicom-station-setup.exe` is the `rds-setup` binary with a
//! zip appended to it and a small fixed-size footer at the very end:
//!
//! ```text
//! [ rds-setup.exe ][ payload.zip ][ "RDSPAY01" | u64 offset | u64 length ]
//! ```
//!
//! Appending rather than `include_bytes!`-ing the payload has one concrete
//! benefit: the first `offset` bytes are exactly the untouched setup binary,
//! so the uninstaller left behind in the install directory is a plain
//! truncated copy of ourselves — a few MB instead of a few hundred.
//!
//! A freshly `cargo build`-ed `rds-setup.exe` has no payload; it then falls
//! back to a `payload/` directory next to the executable, which is handy when
//! working on the installer itself.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Footer magic. Bump the trailing digits if the layout ever changes.
pub const MAGIC: &[u8; 8] = b"RDSPAY01";
/// magic + u64 offset + u64 length.
pub const FOOTER_LEN: usize = 8 + 8 + 8;

/// One file in the payload.
pub struct Entry {
    /// Path relative to the install directory, using `/` separators.
    pub name: String,
    /// Uncompressed size in bytes.
    pub size: u64,
}

pub enum Payload {
    /// Zip appended to our own executable.
    Embedded { exe: PathBuf, offset: u64, len: u64 },
    /// A `payload/` directory next to the executable (developer mode).
    Directory(PathBuf),
}

impl Payload {
    /// Locate the payload for the currently running executable.
    pub fn locate() -> Result<Payload> {
        let exe = std::env::current_exe().context("current_exe")?;
        if let Some((offset, len)) = read_footer(&exe)? {
            return Ok(Payload::Embedded { exe, offset, len });
        }
        let dir = exe
            .parent()
            .map(|p| p.join("payload"))
            .filter(|p| p.is_dir());
        match dir {
            Some(dir) => Ok(Payload::Directory(dir)),
            None => bail!(
                "no payload: this setup binary carries no embedded files and there is no \
                 'payload' directory next to {}.\nBuild a shippable installer with \
                 `cargo run --release --bin rds-pack`.",
                exe.display()
            ),
        }
    }

    /// Byte length of the payload-free prefix of our executable, i.e. how many
    /// bytes to copy to obtain a working (payload-less) uninstaller.
    pub fn base_exe_len(&self) -> Result<u64> {
        Ok(match self {
            Payload::Embedded { offset, .. } => *offset,
            Payload::Directory(_) => std::env::current_exe()?.metadata()?.len(),
        })
    }

    /// List the files without extracting them.
    pub fn entries(&self) -> Result<Vec<Entry>> {
        match self {
            Payload::Embedded { .. } => {
                let mut zip = self.open_zip()?;
                let mut out = Vec::new();
                for i in 0..zip.len() {
                    let f = zip.by_index(i)?;
                    if f.is_file() {
                        out.push(Entry {
                            name: f.name().replace('\\', "/"),
                            size: f.size(),
                        });
                    }
                }
                Ok(out)
            }
            Payload::Directory(dir) => {
                let mut out = Vec::new();
                for e in walkdir::WalkDir::new(dir)
                    .into_iter()
                    .filter_map(|e| e.ok())
                {
                    if !e.file_type().is_file() {
                        continue;
                    }
                    let Ok(rel) = e.path().strip_prefix(dir) else {
                        continue;
                    };
                    out.push(Entry {
                        name: rel.to_string_lossy().replace('\\', "/"),
                        size: e.metadata().map(|m| m.len()).unwrap_or(0),
                    });
                }
                Ok(out)
            }
        }
    }

    /// Total uncompressed size of the payload.
    pub fn total_size(&self) -> Result<u64> {
        Ok(self.entries()?.iter().map(|e| e.size).sum())
    }

    /// Read a single small text file from the payload, if present.
    pub fn read_text(&self, name: &str) -> Option<String> {
        match self {
            Payload::Embedded { .. } => {
                let mut zip = self.open_zip().ok()?;
                let mut f = zip.by_name(name).ok()?;
                let mut s = String::new();
                f.read_to_string(&mut s).ok()?;
                Some(s)
            }
            Payload::Directory(dir) => std::fs::read_to_string(dir.join(name)).ok(),
        }
    }

    /// Write every payload file into `dest`, reporting progress as
    /// `(fraction, current file name)`. Returns the relative paths written.
    pub fn extract_to(
        &self,
        dest: &Path,
        progress: &mut dyn FnMut(f32, &str),
    ) -> Result<Vec<String>> {
        let entries = self.entries()?;
        let total: u64 = entries.iter().map(|e| e.size).sum::<u64>().max(1);
        let mut done: u64 = 0;
        let mut written = Vec::with_capacity(entries.len());
        match self {
            Payload::Embedded { .. } => {
                let mut zip = self.open_zip()?;
                for i in 0..zip.len() {
                    let mut f = zip.by_index(i)?;
                    if !f.is_file() {
                        continue;
                    }
                    let name = f.name().replace('\\', "/");
                    let out = safe_join(dest, &name)?;
                    progress(done as f32 / total as f32, &name);
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)
                            .with_context(|| format!("create {}", parent.display()))?;
                    }
                    let mut w = std::io::BufWriter::new(
                        File::create(&out).with_context(|| format!("write {}", out.display()))?,
                    );
                    std::io::copy(&mut f, &mut w)
                        .with_context(|| format!("write {}", out.display()))?;
                    done += f.size();
                    written.push(name);
                }
            }
            Payload::Directory(dir) => {
                for e in entries {
                    let src = dir.join(&e.name);
                    let out = safe_join(dest, &e.name)?;
                    progress(done as f32 / total as f32, &e.name);
                    if let Some(parent) = out.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::copy(&src, &out)
                        .with_context(|| format!("copy {} -> {}", src.display(), out.display()))?;
                    done += e.size;
                    written.push(e.name);
                }
            }
        }
        progress(1.0, "");
        Ok(written)
    }

    fn open_zip(&self) -> Result<zip::ZipArchive<Section>> {
        let Payload::Embedded { exe, offset, len } = self else {
            bail!("not an embedded payload");
        };
        let section = Section::new(File::open(exe)?, *offset, *len);
        zip::ZipArchive::new(section).context("read embedded payload")
    }
}

/// Reject absolute paths and `..` components before joining onto `dest`.
fn safe_join(dest: &Path, rel: &str) -> Result<PathBuf> {
    let mut out = dest.to_path_buf();
    for part in rel.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." || part.contains(':') {
            bail!("payload contains an unsafe path: {rel}");
        }
        out.push(part);
    }
    Ok(out)
}

/// Read the payload footer of `exe`, if it has one.
pub fn read_footer(exe: &Path) -> Result<Option<(u64, u64)>> {
    let mut f = File::open(exe).with_context(|| format!("open {}", exe.display()))?;
    let file_len = f.metadata()?.len();
    if file_len < FOOTER_LEN as u64 {
        return Ok(None);
    }
    let mut buf = [0u8; FOOTER_LEN];
    f.seek(SeekFrom::End(-(FOOTER_LEN as i64)))?;
    f.read_exact(&mut buf)?;
    if &buf[..8] != MAGIC {
        return Ok(None);
    }
    let offset = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let len = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    if offset + len + FOOTER_LEN as u64 != file_len {
        bail!("corrupt installer payload footer");
    }
    Ok(Some((offset, len)))
}

/// A read-only, seekable window into a larger file.
pub struct Section {
    file: File,
    start: u64,
    len: u64,
    pos: u64,
}

impl Section {
    pub fn new(file: File, start: u64, len: u64) -> Section {
        Section {
            file,
            start,
            len,
            pos: 0,
        }
    }
}

impl Read for Section {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.len.saturating_sub(self.pos);
        if remaining == 0 {
            return Ok(0);
        }
        let want = buf.len().min(remaining as usize);
        self.file.seek(SeekFrom::Start(self.start + self.pos))?;
        let n = self.file.read(&mut buf[..want])?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Section {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::End(n) => self.len as i64 + n,
            SeekFrom::Current(n) => self.pos as i64 + n,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before start of payload",
            ));
        }
        self.pos = (new as u64).min(self.len);
        Ok(self.pos)
    }
}
