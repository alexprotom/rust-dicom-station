//! Weight acquisition and the on-disk converted-weight cache.
//!
//! Every inference engine follows the same path on first use: download the
//! published checkpoint ([`RemoteFile::ensure`]), parse the torch pickle
//! natively (see [`super::pickle`]), and write the tensors it actually needs
//! to a `safetensors` file next to it ([`convert_checkpoint`]). Every later
//! run reads that cache and never touches the network.
//!
//! Everything is pure Rust: TLS via rustls (`ureq`) with the operating
//! system's trust store, so a clinical network doing TLS inspection with its
//! own CA still works; `safetensors` for the cache format.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use super::half::{f16_to_f32, f32_to_f16};
use super::pickle::{PthReader, TensorMeta};
use crate::progress::{ProgressSink, CANCELLED};

/// One file published in a model repository.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    /// File name, used both remotely and in the local cache.
    pub name: &'static str,
    pub url: &'static str,
    /// Published size in bytes; progress display only, and the fallback for
    /// the percentage when the server sends no `Content-Length`.
    pub bytes: u64,
}

impl RemoteFile {
    /// Where the file lives once fetched into `dir`.
    pub fn path_in(&self, dir: &Path) -> PathBuf {
        dir.join(self.name)
    }

    /// True when the file is already in `dir`.
    pub fn is_cached(&self, dir: &Path) -> bool {
        self.path_in(dir).is_file()
    }

    /// Fetch the file into `dir` if it is not already there, and return its
    /// path. Downloads land on a temporary name (`<name>.part`) and are
    /// renamed only once complete, so an interrupted download is never
    /// mistaken for a cached file; the next call continues the `.part`
    /// where it stopped (see [`download_to_file`]).
    pub fn ensure(&self, dir: &Path, sink: &dyn ProgressSink) -> Result<PathBuf> {
        let dest = self.path_in(dir);
        if dest.is_file() {
            return Ok(dest);
        }
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let tmp = dest.with_extension("part");
        download_to_file(self.url, &tmp, self.bytes, self.name, sink)
            .with_context(|| format!("download {}", self.url))?;
        std::fs::rename(&tmp, &dest)?;
        Ok(dest)
    }
}

/// Bytes still to download for the files not yet in `dir`.
pub fn download_needed<'a>(files: impl IntoIterator<Item = &'a RemoteFile>, dir: &Path) -> u64 {
    files
        .into_iter()
        .filter(|f| !f.is_cached(dir))
        .map(|f| f.bytes)
        .sum()
}

/// A dense f32 tensor loaded from a checkpoint or from the cache.
#[derive(Clone)]
pub struct WTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl WTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Element type to store a converted tensor as.
///
/// `F32` reproduces the published checkpoint bit for bit and is what any
/// model validated against a reference implementation must use. `F16` halves
/// the cache on disk at the cost of roughly three decimal digits per weight -
/// fine for a preview, not for numerical equivalence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreDtype {
    F32,
    F16,
}

/// Fetch a URL into `dest`, reporting progress and honoring cancellation,
/// and continue an earlier attempt when one was cut short.
///
/// `dest` is a working name the caller renames once this returns (a
/// `.part` file). A download that fails or is cancelled leaves `dest` as far
/// as it got, together with `<dest>.validator`: the `ETag` (or, failing that,
/// the `Last-Modified` date) the server sent with it. The next call asks for
/// the rest only (`Range: bytes=<len>-`), conditional on that validator
/// (`If-Range`), so a file the server has replaced in the meantime comes
/// back whole (`200`) rather than spliced onto the old start. Without a
/// validator a partial file is not trusted and the download starts over.
///
/// The body must be as long as the server said (`Content-Length`, or the
/// total of a `Content-Range`): a connection closed cleanly in the middle
/// is an error, not a short file a later run would take for the cache.
/// `size_hint` is used for the percentage only; several published sizes are
/// rounded, so it is never checked against.
pub fn download_to_file(
    url: &str,
    dest: &Path,
    size_hint: u64,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<()> {
    let agent = download_agent();
    let validator_path = sidecar(dest, "validator");
    let have = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
    let validator = std::fs::read_to_string(&validator_path)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let resume = match (&validator, have) {
        (Some(v), n) if n > 0 => Some((v.clone(), n)),
        _ => None,
    };
    sink.report(0.0, &format!("Downloading {label}"));
    let mut req = agent.get(url);
    if let Some((v, n)) = &resume {
        req = req.set("Range", &format!("bytes={n}-")).set("If-Range", v);
    }
    let resp = match req.call() {
        Ok(r) => r,
        // The part already holds the whole file, or more than the server's:
        // nothing to continue, so start over.
        Err(ureq::Error::Status(416, _)) if resume.is_some() => {
            let _ = std::fs::remove_file(dest);
            let _ = std::fs::remove_file(&validator_path);
            agent
                .get(url)
                .call()
                .with_context(|| format!("download {url}"))?
        }
        Err(e) => return Err(e).with_context(|| format!("download {url}")),
    };
    let start = match (&resume, resp.status()) {
        (Some((v, n)), 206) => {
            // A partial answer that does not continue exactly this file
            // (no range, another first byte, or a strong ETag that is not
            // the one the part was made from: a cache that ignored
            // `If-Range`) is not appended to. The part is thrown away and
            // the file fetched whole, so a retry does not meet the same
            // answer forever.
            let range = content_range(&resp).filter(|(from, _)| from == n);
            let same = match resp.header("ETag") {
                Some(e) if !e.starts_with("W/") && v.starts_with('"') => e == v,
                _ => true,
            };
            match range {
                Some((_, total)) if same => {
                    let length = resp
                        .header("Content-Length")
                        .and_then(|v| v.parse::<u64>().ok());
                    Some((*n, total.or(length.map(|l| n + l))))
                }
                _ => {
                    drop(resp);
                    let _ = std::fs::remove_file(dest);
                    let _ = std::fs::remove_file(&validator_path);
                    return download_to_file(url, dest, size_hint, label, sink);
                }
            }
        }
        _ => None,
    };
    // Keep what identifies this version of the file before the body is read,
    // so an attempt cut short can be continued.
    let new_validator = resp
        .header("ETag")
        .filter(|e| !e.starts_with("W/"))
        .or_else(|| resp.header("Last-Modified"))
        .map(str::to_string);
    match &new_validator {
        Some(v) => {
            if start.is_none() {
                std::fs::write(&validator_path, v)
                    .with_context(|| format!("write {}", validator_path.display()))?;
            }
        }
        None => {
            let _ = std::fs::remove_file(&validator_path);
        }
    }
    let res = match start {
        Some((from, total)) => write_body(resp, dest, from, total, size_hint, label, sink),
        None => save_body(resp, dest, size_hint, label, sink),
    };
    if res.is_ok() {
        let _ = std::fs::remove_file(&validator_path);
    } else if new_validator.is_none() {
        // Nothing to continue from safely.
        let _ = std::fs::remove_file(dest);
    }
    res
}

/// `<path>.<ext>` beside `path` (the extension appended, not replaced).
fn sidecar(path: &Path, ext: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

/// The first byte and the total length of a `206` answer's
/// `Content-Range: bytes <first>-<last>/<total>`; `None` for the total when
/// the server writes `*`.
fn content_range(resp: &ureq::Response) -> Option<(u64, Option<u64>)> {
    let v = resp.header("Content-Range")?;
    let rest = v.trim().strip_prefix("bytes")?.trim();
    let (range, total) = rest.split_once('/')?;
    let (first, _) = range.split_once('-')?;
    let first = first.trim().parse().ok()?;
    let total = total.trim().parse().ok();
    Some((first, total))
}

/// The HTTP client downloads go through. It keeps its connections open and
/// hands them to the next request for the same host, so fetching many files
/// from one server costs a TLS handshake per connection rather than one per
/// file - the difference between seconds and minutes for a folder of a
/// thousand DICOM slices.
pub fn download_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(60))
        .max_idle_connections_per_host(8)
        .build()
}

/// Stream a URL to `dest` through a given `agent` (one shared by many
/// downloads, see [`download_agent`]), from the start. A cancelled or
/// failed download leaves no file behind; see [`save_response`] for the
/// length check.
pub fn download_with(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    size_hint: u64,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<()> {
    sink.report(0.0, &format!("Downloading {label}"));
    let resp = agent
        .get(url)
        .call()
        .with_context(|| format!("download {url}"))?;
    save_response(resp, dest, size_hint, label, sink)
}

/// Stream a response body to `dest`, reporting progress against its
/// `Content-Length` (or `size_hint`) and honoring cancellation. A body
/// shorter or longer than its `Content-Length` is an error. A cancelled or
/// failed download leaves no file behind.
pub fn save_response(
    resp: ureq::Response,
    dest: &Path,
    size_hint: u64,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<()> {
    let res = save_body(resp, dest, size_hint, label, sink);
    if res.is_err() {
        let _ = std::fs::remove_file(dest);
    }
    res
}

/// A whole body into a new `dest`; on error `dest` holds what arrived.
fn save_body(
    resp: ureq::Response,
    dest: &Path,
    size_hint: u64,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<()> {
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok());
    write_body(resp, dest, 0, total, size_hint, label, sink)
}

/// Write `resp`'s body into `dest` from byte `from` on (truncating it there;
/// `0` starts a new file), and check the file ends at `total` when the
/// server said how long it is.
fn write_body(
    resp: ureq::Response,
    dest: &Path,
    from: u64,
    total: Option<u64>,
    size_hint: u64,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<()> {
    use std::io::Seek;
    let shown_total = total.unwrap_or(size_hint);
    let mut reader = resp.into_reader();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(from == 0)
        .open(dest)
        .with_context(|| format!("create {}", dest.display()))?;
    if from > 0 {
        file.set_len(from)
            .and_then(|()| file.seek(std::io::SeekFrom::Start(from)).map(|_| ()))
            .with_context(|| format!("continue {}", dest.display()))?;
    }
    let mut out = std::io::BufWriter::new(file);
    let mut buf = vec![0u8; 256 * 1024];
    let mut done: u64 = from;
    loop {
        if sink.cancelled() {
            bail!(CANCELLED);
        }
        let n = reader.read(&mut buf).context("download read")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).context("write download")?;
        done += n as u64;
        let frac = (done as f32 / shown_total.max(1) as f32).min(1.0);
        sink.report(
            frac,
            &format!(
                "Downloading {label}: {} / {} MB",
                done / 1_000_000,
                shown_total / 1_000_000
            ),
        );
    }
    out.flush().context("write download")?;
    if let Some(t) = total {
        if done != t {
            bail!("the download of {label} ended at {done} of {t} bytes");
        }
    }
    Ok(())
}

/// Write named tensors to a `safetensors` file, atomically (written to a
/// temporary file, then renamed) so an interrupted conversion never leaves
/// a half-written cache that a later run would trust.
///
/// The temporary file is made in a folder of its own beside the cache,
/// `<name>.converting`, which the next conversion of the same file empties
/// first: a conversion killed half way (hundreds of megabytes) is not left
/// in the model folder under a random name nothing would ever reuse.
///
/// The tensors are borrowed, so a caller that keeps using them afterwards
/// (as every engine does on first use) pays for no copy; the file is
/// streamed out a tensor at a time, so the only extra memory is the bytes
/// of the tensor being written, never a second copy of the whole model.
pub fn save_safetensors<'a>(
    path: &Path,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize], &'a [f32])>,
    dtype: StoreDtype,
) -> Result<()> {
    let views = tensors
        .into_iter()
        .map(|(name, shape, data)| (name, PackView { shape, data, dtype }));
    let staging = sidecar(path, "converting");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).with_context(|| format!("create {}", staging.display()))?;
    let name = path
        .file_name()
        .with_context(|| format!("{} names no file", path.display()))?;
    let staged = staging.join(name);
    let written = safetensors::tensor::serialize_to_file(views, None, &staged)
        .with_context(|| format!("write {}", path.display()))
        .and_then(|()| {
            std::fs::rename(&staged, path).with_context(|| format!("write {}", path.display()))
        });
    let _ = std::fs::remove_dir_all(&staging);
    written
}

/// One tensor as `safetensors` writes it: its little-endian bytes made when
/// the writer asks for them, one tensor at a time.
struct PackView<'a> {
    shape: &'a [usize],
    data: &'a [f32],
    dtype: StoreDtype,
}

impl safetensors::tensor::View for PackView<'_> {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        match self.dtype {
            StoreDtype::F32 => safetensors::tensor::Dtype::F32,
            StoreDtype::F16 => safetensors::tensor::Dtype::F16,
        }
    }

    fn shape(&self) -> &[usize] {
        self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Owned(pack_le(self.data, self.dtype))
    }

    fn data_len(&self) -> usize {
        self.data.len()
            * match self.dtype {
                StoreDtype::F32 => 4,
                StoreDtype::F16 => 2,
            }
    }
}

/// [`save_safetensors`] for a loaded tensor map.
pub fn save_tensor_map(
    path: &Path,
    tensors: &HashMap<String, WTensor>,
    dtype: StoreDtype,
) -> Result<()> {
    save_safetensors(
        path,
        tensors
            .iter()
            .map(|(k, t)| (k.as_str(), t.shape.as_slice(), t.data.as_slice())),
        dtype,
    )
}

/// Little-endian bytes of a tensor, written straight into a sized buffer
/// rather than pushed four bytes at a time.
fn pack_le(data: &[f32], dtype: StoreDtype) -> Vec<u8> {
    match dtype {
        StoreDtype::F32 => {
            let mut b = vec![0u8; data.len() * 4];
            for (dst, v) in b.as_chunks_mut::<4>().0.iter_mut().zip(data) {
                *dst = v.to_le_bytes();
            }
            b
        }
        StoreDtype::F16 => {
            let mut b = vec![0u8; data.len() * 2];
            for (dst, v) in b.as_chunks_mut::<2>().0.iter_mut().zip(data) {
                *dst = f32_to_f16(*v).to_le_bytes();
            }
            b
        }
    }
}

/// One tensor's entry in a `safetensors` header.
struct Entry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    /// Byte range in the file (the header's offsets plus the header).
    begin: u64,
    end: u64,
}

/// The tensor entries of a `safetensors` file, read from its header alone.
///
/// The format is an 8-byte little-endian header length, the header (JSON:
/// name to dtype, shape and `data_offsets` into the data that follows), then
/// the data. Every entry is checked against the file's length and its own
/// shape, the checks `SafeTensors::deserialize` makes, without reading the
/// data.
fn read_header(file: &mut std::fs::File, path: &Path) -> Result<Vec<Entry>> {
    let file_len = file.metadata()?.len();
    let mut n = [0u8; 8];
    file.read_exact(&mut n)
        .with_context(|| format!("parse {}: too short", path.display()))?;
    let n = u64::from_le_bytes(n);
    if n > 100_000_000 || 8 + n > file_len {
        bail!("parse {}: bad header length {n}", path.display());
    }
    let mut header = vec![0u8; n as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("parse {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("parse {}: header", path.display()))?;
    let map = json
        .as_object()
        .with_context(|| format!("parse {}: header is not a map", path.display()))?;
    let data_start = 8 + n;
    let mut out = Vec::with_capacity(map.len());
    for (name, v) in map {
        if name == "__metadata__" {
            continue;
        }
        let bad = || format!("parse {}: tensor {name}", path.display());
        let dtype = v["dtype"].as_str().with_context(bad)?.to_string();
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .with_context(bad)?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<_>>()
            .with_context(bad)?;
        let offs = v["data_offsets"].as_array().with_context(bad)?;
        let (a, b) = match offs.as_slice() {
            [a, b] => (a.as_u64().with_context(bad)?, b.as_u64().with_context(bad)?),
            _ => bail!(bad()),
        };
        let elem = match dtype.as_str() {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "F32" | "I32" | "U32" => 4,
            "F64" | "I64" | "U64" => 8,
            _ => 0,
        };
        let numel: u64 = shape.iter().map(|&d| d as u64).product();
        if b < a || data_start + b > file_len || (elem > 0 && b - a != numel * elem) {
            bail!(
                "parse {}: tensor {name} has inconsistent offsets",
                path.display()
            );
        }
        out.push(Entry {
            name: name.clone(),
            dtype,
            shape,
            begin: data_start + a,
            end: data_start + b,
        });
    }
    Ok(out)
}

/// Read a `safetensors` cache, widening half-precision tensors to `f32`.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, WTensor>> {
    load_safetensors_where(path, &|_| true)
}

/// [`load_safetensors`] for the tensors whose name `keep` accepts: the rest
/// of the file is never read (SegVol's text tower is a few of the 724 MB).
///
/// Each tensor is read from its own place in the file straight into its
/// `f32` buffer, a megabyte at a time, so loading costs the model's size in
/// memory and not twice that (the file's bytes and then the decoded copy).
/// The tensors are independent and read on every core.
pub fn load_safetensors_where(
    path: &Path,
    keep: &(dyn Fn(&str) -> bool + Sync),
) -> Result<HashMap<String, WTensor>> {
    use rayon::prelude::*;
    let mut file = std::fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let entries = read_header(&mut file, path)?;
    drop(file);
    let decoded: Vec<Option<(String, WTensor)>> = entries
        .into_par_iter()
        .filter(|e| keep(&e.name))
        .map(|e| {
            let width = match e.dtype.as_str() {
                "F32" => 4,
                "F16" => 2,
                // Integer buffers a published safetensors file may carry
                // beside the weights (`num_batches_tracked`) are not
                // weights; nothing reads them.
                "I64" | "I32" | "U8" | "BOOL" => return Ok(None),
                other => bail!("cached tensor {} has unsupported dtype {other}", e.name),
            };
            let data = read_tensor(path, e.begin, e.end, width)
                .with_context(|| format!("read {}: tensor {}", path.display(), e.name))?;
            Ok(Some((
                e.name,
                WTensor {
                    shape: e.shape,
                    data,
                },
            )))
        })
        .collect::<Result<_>>()?;
    Ok(decoded.into_iter().flatten().collect())
}

/// Bytes `begin..end` of `path` as little-endian `f32` (`width` 4) or
/// binary16 (`width` 2) values, widened to `f32`.
fn read_tensor(path: &Path, begin: u64, end: u64, width: usize) -> Result<Vec<f32>> {
    use std::io::Seek;
    const CHUNK: usize = 1 << 20;
    let mut file = std::fs::File::open(path)?;
    file.seek(std::io::SeekFrom::Start(begin))?;
    let len = (end - begin) as usize;
    let mut out = Vec::with_capacity(len / width);
    let mut buf = vec![0u8; CHUNK.min(len)];
    let mut left = len;
    while left > 0 {
        let k = left.min(CHUNK);
        let chunk = &mut buf[..k];
        file.read_exact(chunk)?;
        if width == 4 {
            out.extend(
                chunk
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| f32::from_le_bytes(c)),
            );
        } else {
            out.extend(
                chunk
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| f16_to_f32(u16::from_le_bytes(c))),
            );
        }
        left -= k;
    }
    Ok(out)
}

/// Which tensors of a checkpoint to convert, and under what names.
pub struct ConvertSpec<'a> {
    /// Entry of the checkpoint's root dict holding the state dict; empty when
    /// the root *is* the state dict.
    pub top_key: &'a str,
    /// Keep this tensor? Called with the checkpoint's own name.
    pub keep: &'a (dyn Fn(&str, &TensorMeta) -> bool + Sync),
    /// The name the tensor is cached under.
    pub rename: &'a (dyn Fn(&str) -> String + Sync),
    /// Interpolated into progress messages.
    pub label: &'a str,
}

/// Read every tensor `spec` keeps out of the torch checkpoint at `pth`, write
/// them to the `safetensors` cache at `cache`, and hand them back so the
/// first run does not have to read the cache it just wrote.
///
/// Always `f32`: every engine here is validated tensor for tensor against its
/// reference implementation, so the cache must reproduce the checkpoint.
pub fn convert_checkpoint(
    pth: &Path,
    cache: &Path,
    spec: &ConvertSpec,
    sink: &dyn ProgressSink,
) -> Result<HashMap<String, WTensor>> {
    let mut reader = PthReader::open(pth, spec.top_key)
        .with_context(|| format!("read checkpoint {}", pth.display()))?;
    let metas: Vec<(String, TensorMeta)> = std::mem::take(&mut reader.tensors)
        .into_iter()
        .filter(|(name, meta)| (spec.keep)(name, meta))
        .collect();
    if metas.is_empty() {
        bail!("checkpoint {}: no usable tensors", pth.display());
    }
    let n = metas.len();
    let mut out = HashMap::with_capacity(n);
    for (i, (name, meta)) in metas.iter().enumerate() {
        if sink.cancelled() {
            bail!(CANCELLED);
        }
        sink.report(
            i as f32 / n as f32,
            &format!("Converting weights ({}): {}/{n}", spec.label, i + 1),
        );
        let data = reader
            .read_f32(meta)
            .with_context(|| format!("read tensor {name}"))?;
        out.insert(
            (spec.rename)(name),
            WTensor {
                shape: meta.shape.clone(),
                data,
            },
        );
    }
    if let Some(dir) = cache.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    sink.report(1.0, &format!("Writing the weight cache ({})", spec.label));
    save_tensor_map(cache, &out, StoreDtype::F32)
        .with_context(|| format!("write {}", cache.display()))?;
    Ok(out)
}

/// The whole first-use path: load the converted cache when it exists,
/// otherwise obtain the checkpoint (`fetch`, which downloads on demand),
/// convert it and cache the result.
pub fn ensure_converted(
    cache: &Path,
    spec: &ConvertSpec,
    fetch: impl FnOnce() -> Result<PathBuf>,
    sink: &dyn ProgressSink,
) -> Result<HashMap<String, WTensor>> {
    if cache.is_file() {
        sink.report(0.0, &format!("Loading weights ({})", spec.label));
        return load_safetensors(cache);
    }
    let pth = fetch()?;
    convert_checkpoint(&pth, cache, spec, sink)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn borrowed(
        t: &[(String, Vec<usize>, Vec<f32>)],
    ) -> impl Iterator<Item = (&str, &[usize], &[f32])> {
        t.iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
    }

    fn sample() -> Vec<(String, Vec<usize>, Vec<f32>)> {
        vec![
            (
                "block.weight".to_string(),
                vec![2, 3],
                vec![1.0, -2.0, 0.5, 0.25, -0.125, 65504.0],
            ),
            ("block.bias".to_string(), vec![2], vec![0.0, -0.0]),
        ]
    }

    #[test]
    fn f32_cache_round_trips_exactly() {
        let dir = std::env::temp_dir().join("rds_nn_cache_f32");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let t = sample();
        save_safetensors(&path, borrowed(&t), StoreDtype::F32).unwrap();
        let back = load_safetensors(&path).unwrap();
        assert_eq!(back.len(), 2);
        for (name, shape, data) in &t {
            let got = &back[name];
            assert_eq!(&got.shape, shape);
            assert_eq!(&got.data, data, "{name}");
            assert_eq!(got.numel(), data.len());
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f16_cache_round_trips_within_half_precision() {
        let dir = std::env::temp_dir().join("rds_nn_cache_f16");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let t = sample();
        save_safetensors(&path, borrowed(&t), StoreDtype::F16).unwrap();
        let back = load_safetensors(&path).unwrap();
        // every sample value is exactly representable in binary16
        for (name, _, data) in &t {
            assert_eq!(&back[name].data, data, "{name}");
        }
        // and the file really is half the size of the f32 one
        let f16_len = std::fs::metadata(&path).unwrap().len();
        save_safetensors(&path, borrowed(&t), StoreDtype::F32).unwrap();
        let f32_len = std::fs::metadata(&path).unwrap().len();
        assert!(f16_len < f32_len, "{f16_len} !< {f32_len}");
        std::fs::remove_file(&path).ok();
    }

    /// A server for one file on the loopback, answering each request with
    /// the next of `script`: `(status, body range, ETag, cut)`, where `cut`
    /// closes the connection after that many bytes of the body.
    fn serve(
        file: &'static [u8],
        script: Vec<(u16, std::ops::Range<usize>, &'static str, Option<usize>)>,
    ) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.bin", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            use std::io::BufRead;
            let mut seen = Vec::new();
            for (status, range, etag, cut) in script {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut head = String::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    head.push_str(&line);
                }
                seen.push(head);
                let body = &file[range.clone()];
                let mut out = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nETag: {etag}\r\nConnection: close\r\n",
                    body.len()
                );
                if status == 206 {
                    out.push_str(&format!(
                        "Content-Range: bytes {}-{}/{}\r\n",
                        range.start,
                        range.end - 1,
                        file.len()
                    ));
                }
                out.push_str("\r\n");
                stream.write_all(out.as_bytes()).unwrap();
                let send = cut.unwrap_or(body.len()).min(body.len());
                stream.write_all(&body[..send]).unwrap();
            }
            seen
        });
        (url, handle)
    }

    #[test]
    fn a_cut_download_is_an_error_and_continues_where_it_stopped() {
        static FILE: [u8; 1000] = {
            let mut f = [0u8; 1000];
            let mut i = 0;
            while i < 1000 {
                f[i] = (i * 7 % 251) as u8;
                i += 1;
            }
            f
        };
        let dir = std::env::temp_dir().join(format!("rds_dl_resume_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let part = dir.join("model.part");
        let quiet = crate::progress::Quiet;
        // Cut after 400 bytes, then continued from there.
        let (url, server) = serve(
            &FILE,
            vec![
                (200, 0..1000, "\"v1\"", Some(400)),
                (206, 400..1000, "\"v1\"", None),
            ],
        );
        // Whether the HTTP client or the length check notices first, it is
        // an error and not a short file.
        assert!(download_to_file(&url, &part, 1000, "model", &quiet).is_err());
        assert_eq!(
            std::fs::metadata(&part).unwrap().len(),
            400,
            "kept to continue"
        );
        download_to_file(&url, &part, 1000, "model", &quiet).unwrap();
        assert_eq!(std::fs::read(&part).unwrap(), FILE.to_vec());
        assert!(!sidecar(&part, "validator").exists());
        let heads = server.join().unwrap();
        assert!(!heads[0].contains("Range"), "{}", heads[0]);
        assert!(heads[1].contains("Range: bytes=400-"), "{}", heads[1]);
        assert!(heads[1].contains("If-Range: \"v1\""), "{}", heads[1]);
        // A file replaced on the server answers the range with the whole of
        // itself, which replaces what was kept.
        std::fs::write(&part, &FILE[..300]).unwrap();
        std::fs::write(sidecar(&part, "validator"), "\"v0\"").unwrap();
        let (url, server) = serve(&FILE, vec![(200, 0..1000, "\"v1\"", None)]);
        download_to_file(&url, &part, 1000, "model", &quiet).unwrap();
        assert_eq!(std::fs::read(&part).unwrap(), FILE.to_vec());
        server.join().unwrap();
        // A partial answer of another version (a cache that ignored
        // If-Range) or from another byte is not appended to: the part goes
        // and the file comes whole at once.
        for (etag, from) in [("\"v1\"", 300), ("\"v0\"", 200)] {
            std::fs::write(&part, &FILE[..300]).unwrap();
            std::fs::write(sidecar(&part, "validator"), "\"v0\"").unwrap();
            let (url, server) = serve(
                &FILE,
                vec![
                    (206, from..1000, etag, None),
                    (200, 0..1000, "\"v1\"", None),
                ],
            );
            download_to_file(&url, &part, 1000, "model", &quiet).unwrap();
            assert_eq!(std::fs::read(&part).unwrap(), FILE.to_vec());
            let heads = server.join().unwrap();
            assert!(heads[0].contains("Range: bytes=300-"), "{}", heads[0]);
            assert!(!heads[1].contains("Range"), "{}", heads[1]);
        }
        // Without a validator a partial file is not trusted.
        std::fs::write(&part, &FILE[..300]).unwrap();
        let (url, server) = serve(&FILE, vec![(200, 0..1000, "\"v1\"", None)]);
        download_to_file(&url, &part, 1000, "model", &quiet).unwrap();
        assert_eq!(std::fs::read(&part).unwrap(), FILE.to_vec());
        assert!(!server.join().unwrap()[0].contains("Range"));
        // The plain path deletes what a cut download left.
        let (url, server) = serve(&FILE, vec![(200, 0..1000, "\"v1\"", Some(10))]);
        let whole = dir.join("plain.bin");
        assert!(download_with(&download_agent(), &url, &whole, 1000, "plain", &quiet).is_err());
        assert!(!whole.exists());
        server.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_is_atomic() {
        // The temporary file must not survive a successful save, or a later
        // run could pick it up as a stale cache.
        let dir = std::env::temp_dir().join("rds_nn_cache_atomic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let t = sample();
        save_safetensors(&path, borrowed(&t), StoreDtype::F32).unwrap();
        assert!(path.is_file());
        assert!(!path.with_extension("safetensors.tmp").exists());
        // Nor does the folder it was written in, and a leftover of a
        // conversion killed half way is cleared by the next one.
        let staging = sidecar(&path, "converting");
        assert!(!staging.exists());
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join(".tmpKILLED"), b"half").unwrap();
        save_safetensors(&path, borrowed(&t), StoreDtype::F32).unwrap();
        assert!(!staging.exists());
        assert_eq!(
            load_safetensors(&path).unwrap()["block.weight"].data,
            t[0].2
        );
        std::fs::remove_file(&path).ok();
    }
}
