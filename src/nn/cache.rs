//! Weight acquisition and the on-disk converted-weight cache.
//!
//! Both inference engines follow the same path on first use: download the
//! published checkpoint, parse the torch pickle natively (see
//! [`super::pickle`]), and write the tensors we actually need to a
//! `safetensors` file next to it. Every later run memory-maps that cache and
//! never touches the network.
//!
//! Everything is pure Rust: TLS via rustls (`ureq`) with the operating
//! system's trust store, so a clinical network doing TLS inspection with its
//! own CA still works; `safetensors` for the cache format.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use super::half::{f16_to_f32, f32_to_f16};

/// Progress reporting + cancellation for long operations.
pub trait ProgressSink: Sync {
    fn report(&self, frac: f32, msg: &str);
    fn cancelled(&self) -> bool {
        false
    }
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
/// the cache on disk at the cost of roughly three decimal digits per weight —
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
    sink.report(0.0, &format!("Downloading {label}…"));
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
            bail!("cancelled");
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
pub fn save_safetensors(
    path: &Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    dtype: StoreDtype,
) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};
    let mut byte_bufs: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());
    for (_, _, data) in tensors {
        let mut b = Vec::with_capacity(data.len() * dtype_size(dtype));
        match dtype {
            StoreDtype::F32 => {
                for v in data {
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
            StoreDtype::F16 => {
                for v in data {
                    b.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
                }
            }
        }
        byte_bufs.push(b);
    }
    let st_dtype = match dtype {
        StoreDtype::F32 => Dtype::F32,
        StoreDtype::F16 => Dtype::F16,
    };
    let mut views: Vec<(&str, TensorView)> = Vec::with_capacity(tensors.len());
    for ((name, shape, _), bytes) in tensors.iter().zip(byte_bufs.iter()) {
        views.push((
            name.as_str(),
            TensorView::new(st_dtype, shape.clone(), bytes).context("safetensors view")?,
        ));
    }
    let ser = safetensors::tensor::serialize(views, None).context("safetensors serialize")?;
    let tmp = path.with_extension("safetensors.tmp");
    std::fs::write(&tmp, ser)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn dtype_size(d: StoreDtype) -> usize {
    match d {
        StoreDtype::F32 => 4,
        StoreDtype::F16 => 2,
    }
}

/// Read a `safetensors` cache, widening half-precision tensors to `f32`.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, WTensor>> {
    use safetensors::tensor::Dtype;
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse {}", path.display()))?;
    let mut out = HashMap::new();
    for (name, view) in st.tensors() {
        let raw = view.data();
        let data = match view.dtype() {
            Dtype::F32 => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Dtype::F16 => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            other => bail!("cached tensor {name} has unsupported dtype {other:?}"),
        };
        out.insert(
            name.to_string(),
            WTensor {
                shape: view.shape().to_vec(),
                data,
            },
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        save_safetensors(&path, &t, StoreDtype::F32).unwrap();
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
        save_safetensors(&path, &t, StoreDtype::F16).unwrap();
        let back = load_safetensors(&path).unwrap();
        // every sample value is exactly representable in binary16
        for (name, _, data) in &t {
            assert_eq!(&back[name].data, data, "{name}");
        }
        // and the file really is half the size of the f32 one
        let f16_len = std::fs::metadata(&path).unwrap().len();
        save_safetensors(&path, &t, StoreDtype::F32).unwrap();
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
        save_safetensors(&path, &sample(), StoreDtype::F32).unwrap();
        assert!(path.is_file());
        assert!(!path.with_extension("safetensors.tmp").exists());
        std::fs::remove_file(&path).ok();
    }
}
