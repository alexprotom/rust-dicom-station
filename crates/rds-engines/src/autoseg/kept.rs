//! Networks kept from one run to the next while a caller asks for it.
//!
//! An automatic model reads each part's folds from the converted cache,
//! assembles the networks and, on the GPU, uploads them: a second or two per
//! part, and on a model in five parts most of a gigabyte. A run over the
//! phases of a 4D group does that once per phase for the same networks.
//! While a [`KeepNetworks`] is alive, the networks a run builds are kept,
//! keyed by what they were built from (the sub-model, its folder, the folds,
//! the device), and the next run that asks for the same ones starts at the
//! image; when the last one is dropped, so are the networks. Without one, a
//! run builds each part just before its windows and drops it after them, so
//! at most one part's networks are resident at a time.
//!
//! What is kept is bounded by [`KEEP_BYTES`] of weights, the most recently
//! used kept first: the five parts of `total` fit, so a 4D run over them
//! pays the loading once; a model larger than that is rebuilt per run as
//! before rather than pinned on a GPU that may not have the room.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;

use super::net::UNet;

/// What a part's networks were built from. Two runs that agree on all of
/// it can share them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct NetKey {
    /// The sub-model (`ModelSpec::key`).
    pub spec: &'static str,
    /// Its folder: another model folder holds other weights.
    pub dir: PathBuf,
    /// How many folds were loaded.
    pub folds: u8,
    /// Where they run: the CPU, or the inference GPU's description.
    pub device: String,
}

/// One part's networks, one per fold, on the device they run on.
pub(super) enum Nets {
    Cpu(Vec<UNet>),
    #[cfg(feature = "gpu")]
    Gpu(Vec<super::gpu::GpuNet>),
}

/// A part ready to run.
pub(super) struct PartNets {
    pub classes: usize,
    pub nets: Nets,
    /// The weights' size in bytes, for [`KEEP_BYTES`].
    pub bytes: usize,
}

/// The most weight bytes kept between runs, all models together.
pub const KEEP_BYTES: usize = 1 << 30;

/// The keepers alive and what they keep, under one lock: a run that builds
/// while the last keeper goes cannot slip its networks in after the
/// release.
struct Kept {
    keepers: usize,
    /// Most recently used last.
    nets: Vec<(NetKey, Arc<PartNets>)>,
}

static KEPT: Mutex<Kept> = Mutex::new(Kept {
    keepers: 0,
    nets: Vec::new(),
});

/// While one of these is alive, automatic models keep the networks they
/// build for the next run (see the module notes). Hold one across a loop
/// that runs the same model several times: the phases of a 4D group, the
/// two images of an evaluation.
#[must_use = "the networks are kept only while this is alive"]
pub struct KeepNetworks {
    _private: (),
}

impl KeepNetworks {
    pub fn new() -> KeepNetworks {
        kept().keepers += 1;
        KeepNetworks { _private: () }
    }
}

impl Default for KeepNetworks {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for KeepNetworks {
    fn drop(&mut self) {
        let mut kept = kept();
        kept.keepers = kept.keepers.saturating_sub(1);
        if kept.keepers == 0 {
            // Dropped outside the lock: freeing GPU buffers can take a while.
            let gone = std::mem::take(&mut kept.nets);
            drop(kept);
            drop(gone);
        }
    }
}

/// Drop every kept network now (the model folder changed, a model was
/// removed, memory is short). A run holding one keeps it until it ends.
pub fn release_networks() {
    let gone = std::mem::take(&mut kept().nets);
    drop(gone);
}

fn kept() -> std::sync::MutexGuard<'static, Kept> {
    KEPT.lock().unwrap_or_else(|p| p.into_inner())
}

/// The networks kept for `key`, or `build`'s, kept when someone asks for
/// that.
pub(super) fn get_or_build(
    key: &NetKey,
    build: impl FnOnce() -> Result<PartNets>,
) -> Result<Arc<PartNets>> {
    {
        let mut kept = kept();
        if let Some(i) = kept.nets.iter().position(|(k, _)| k == key) {
            let entry = kept.nets.remove(i);
            let n = entry.1.clone();
            kept.nets.push(entry);
            return Ok(n);
        }
    }
    let built = Arc::new(build()?);
    let mut kept = kept();
    if kept.keepers > 0 && built.bytes <= KEEP_BYTES {
        // Another run may have built the same meanwhile; keep one.
        if let Some((_, n)) = kept.nets.iter().find(|(k, _)| k == key) {
            return Ok(n.clone());
        }
        kept.nets.push((key.clone(), built.clone()));
        let mut total: usize = kept.nets.iter().map(|(_, n)| n.bytes).sum();
        let mut evicted = Vec::new();
        while total > KEEP_BYTES && kept.nets.len() > 1 {
            let (_, n) = kept.nets.remove(0);
            total -= n.bytes;
            evicted.push(n);
        }
        drop(kept);
        drop(evicted);
    }
    Ok(built)
}

/// Kept entries whose sub-model key starts with `prefix` (tests).
#[cfg(test)]
fn kept_count(prefix: &str) -> usize {
    kept()
        .nets
        .iter()
        .filter(|(k, _)| k.spec.starts_with(prefix))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(spec: &'static str) -> NetKey {
        NetKey {
            spec,
            dir: PathBuf::from("/models/totalsegmentator"),
            folds: 1,
            device: "CPU".into(),
        }
    }

    fn empty() -> Result<PartNets> {
        Ok(PartNets {
            classes: 2,
            nets: Nets::Cpu(Vec::new()),
            bytes: 1,
        })
    }

    fn huge() -> Result<PartNets> {
        Ok(PartNets {
            classes: 2,
            nets: Nets::Cpu(Vec::new()),
            bytes: KEEP_BYTES / 2 + 1,
        })
    }

    /// Kept while a keeper lives, rebuilt otherwise, all dropped with the
    /// last keeper. One test, because the keepers are process-wide.
    #[test]
    fn networks_are_kept_only_while_someone_asks() {
        let a = get_or_build(&key("kept_test_a"), empty).unwrap();
        let b = get_or_build(&key("kept_test_a"), empty).unwrap();
        assert!(!Arc::ptr_eq(&a, &b), "nothing is kept without a keeper");
        let outer = KeepNetworks::new();
        let a = get_or_build(&key("kept_test_a"), empty).unwrap();
        let inner = KeepNetworks::new();
        let b = get_or_build(&key("kept_test_a"), || panic!("built twice")).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let c = get_or_build(&key("kept_test_c"), empty).unwrap();
        assert!(!Arc::ptr_eq(&a, &c));
        assert_eq!(kept_count("kept_test"), 2);
        drop(inner);
        assert_eq!(kept_count("kept_test"), 2, "one keeper is left");
        drop(outer);
        assert_eq!(kept_count("kept_test"), 0);
        // A run that holds its networks keeps them past the release.
        assert_eq!(c.classes, 2);

        // Past the byte budget the least recently used goes first.
        let keep = KeepNetworks::new();
        let x = get_or_build(&key("kept_test_x"), huge).unwrap();
        let _y = get_or_build(&key("kept_test_y"), huge).unwrap();
        assert_eq!(kept_count("kept_test_x"), 0, "x was the oldest");
        assert_eq!(kept_count("kept_test_y"), 1);
        let x2 = get_or_build(&key("kept_test_x"), huge).unwrap();
        assert!(!Arc::ptr_eq(&x, &x2), "x was built again");
        assert_eq!(kept_count("kept_test_y"), 0, "and y made room for it");
        drop(keep);
        assert_eq!(kept_count("kept_test"), 0);
    }
}
