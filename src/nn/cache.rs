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
    /// path. Downloads land on a temporary name and are renamed only once
    /// complete, so an interrupted download is never mistaken for a cached
    /// file.
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

/// Stream a URL to `dest`, reporting progress and honoring cancellation.
///
/// `size_hint` is used for the percentage only when the server sends no
/// `Content-Length`. `label` is interpolated into the progress messages as
/// "Downloading {label}: 12 / 724 MB". A cancelled or failed download leaves
/// no file behind.
pub fn download_to_file(
    url: &str,
    dest: &Path,
    size_hint: u64,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<()> {
    sink.report(0.0, &format!("Downloading {label}"));
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(60))
        .build();
    let resp = agent
        .get(url)
        .call()
        .with_context(|| format!("download {url}"))?;
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(size_hint);
    let mut reader = resp.into_reader();
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?,
    );
    let mut buf = vec![0u8; 256 * 1024];
    let mut done: u64 = 0;
    loop {
        if sink.cancelled() {
            drop(out);
            let _ = std::fs::remove_file(dest);
            bail!(CANCELLED);
        }
        let n = reader.read(&mut buf).context("download read")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).context("write download")?;
        done += n as u64;
        let frac = (done as f32 / total.max(1) as f32).min(1.0);
        sink.report(
            frac,
            &format!(
                "Downloading {label}: {} / {} MB",
                done / 1_000_000,
                total / 1_000_000
            ),
        );
    }
    out.flush().ok();
    Ok(())
}

/// Write named tensors to a `safetensors` file, atomically (write to a
/// temporary file, then rename) so an interrupted conversion never leaves a
/// half-written cache that a later run would trust.
///
/// The tensors are borrowed, so a caller that keeps using them afterwards
/// (as every engine does on first use) pays for no copy.
pub fn save_safetensors<'a>(
    path: &Path,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize], &'a [f32])>,
    dtype: StoreDtype,
) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};
    let tensors: Vec<(&str, &[usize], &[f32])> = tensors.into_iter().collect();
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, data)| pack_le(data, dtype))
        .collect();
    let st_dtype = match dtype {
        StoreDtype::F32 => Dtype::F32,
        StoreDtype::F16 => Dtype::F16,
    };
    let mut views: Vec<(&str, TensorView)> = Vec::with_capacity(tensors.len());
    for ((name, shape, _), bytes) in tensors.iter().zip(byte_bufs.iter()) {
        views.push((
            name,
            TensorView::new(st_dtype, shape.to_vec(), bytes).context("safetensors view")?,
        ));
    }
    let ser = safetensors::tensor::serialize(views, None).context("safetensors serialize")?;
    let tmp = path.with_extension("safetensors.tmp");
    std::fs::write(&tmp, ser)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
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

/// Read a `safetensors` cache, widening half-precision tensors to `f32`.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, WTensor>> {
    use rayon::prelude::*;
    use safetensors::tensor::Dtype;
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse {}", path.display()))?;
    // Decoding is a plain byte reinterpretation per tensor; the tensors are
    // independent, so the (up to 724 MB) cache is unpacked on every core.
    let views = st.tensors();
    views
        .into_par_iter()
        .map(|(name, view)| {
            let raw = view.data();
            let data = match view.dtype() {
                Dtype::F32 => raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| f32::from_le_bytes(c))
                    .collect(),
                Dtype::F16 => raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| f16_to_f32(u16::from_le_bytes(c)))
                    .collect(),
                other => bail!("cached tensor {name} has unsupported dtype {other:?}"),
            };
            Ok((
                name.to_string(),
                WTensor {
                    shape: view.shape().to_vec(),
                    data,
                },
            ))
        })
        .collect()
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
        std::fs::remove_file(&path).ok();
    }
}
