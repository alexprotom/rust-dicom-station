//! Folders of fine-tuned TotalSynth checkpoints the user adds, run like the
//! published apps.
//!
//! KonfAI's `fine-tune` (and its training) writes checkpoints of the same
//! network, `CV_0.pt` to `CV_4.pt` or any other `.pt` name; a folder of them
//! is one app. The checkpoints are cross-validation members of one model,
//! averaged like the published five. The recipe is the published apps'
//! ([`super`]); a model trained on another one is not something this
//! engine can tell from its weights.
//!
//! The folder is only read: the converted caches go to the model folder's
//! `synth/custom/<key>/`. Registered apps live for the rest of the process
//! (the registry hands out `'static` rows, as `autoseg::custom` does).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use anyhow::{bail, Context, Result};

use super::{layout, App, CustomApp};
use crate::nn::pickle::PthReader;

/// The checkpoints in `dir`: `CV_<k>.pt` in the order of `k` when the
/// folder holds those, any other `*.pt` by name after them.
pub fn checkpoints(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut cv: Vec<(usize, PathBuf)> = Vec::new();
    let mut other: Vec<PathBuf> = Vec::new();
    for e in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let p = e?.path();
        if !p.is_file() || p.extension().and_then(|x| x.to_str()) != Some("pt") {
            continue;
        }
        let stem = p
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        match stem
            .strip_prefix("CV_")
            .and_then(|k| k.parse::<usize>().ok())
        {
            Some(k) => cv.push((k, p)),
            None => other.push(p),
        }
    }
    cv.sort();
    other.sort();
    let all: Vec<PathBuf> = cv.into_iter().map(|(_, p)| p).chain(other).collect();
    if all.is_empty() {
        bail!("{} holds no checkpoint (*.pt)", dir.display());
    }
    Ok(all)
}

/// Read and check a folder: its checkpoints, the first of which must be a
/// TotalSynth network.
pub fn describe(dir: &Path) -> Result<(String, Vec<PathBuf>)> {
    let cps = checkpoints(dir)?;
    let r = PthReader::open(&cps[0], "?Model.*")
        .with_context(|| format!("read checkpoint {}", cps[0].display()))?;
    if !layout::recognised(r.tensors.iter().map(|(k, _)| k.as_str())) {
        bail!(
            "{} is not a TotalSynth checkpoint (no UNet++ with a ResNet encoder in it)",
            cps[0].display()
        );
    }
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "checkpoints".to_string());
    Ok((name, cps))
}

/// The registry key of a folder: `sct_custom_` and its name in lower case.
pub fn key_of(name: &str) -> String {
    let mut k = String::from("sct_custom_");
    for c in name.chars() {
        k.push(if c.is_ascii_alphanumeric() {
            c.to_ascii_lowercase()
        } else {
            '_'
        });
    }
    k
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

struct Registry {
    /// Every folder built in this process, by path.
    built: HashMap<PathBuf, &'static CustomApp>,
    /// The ones registered now, in order.
    live: Vec<&'static CustomApp>,
}

static REGISTRY: RwLock<Option<Registry>> = RwLock::new(None);

/// Make `dirs` the registered checkpoint folders (replacing the previous
/// set). Returns, per folder, its app key or why it was refused.
pub fn register(dirs: &[PathBuf]) -> Vec<(PathBuf, Result<&'static str>)> {
    let mut out = Vec::new();
    let Ok(mut reg) = REGISTRY.write() else {
        return out;
    };
    let reg = reg.get_or_insert_with(|| Registry {
        built: HashMap::new(),
        live: Vec::new(),
    });
    reg.live.clear();
    for d in dirs {
        let res = describe(d).map(|(name, cps)| {
            if let Some(a) = reg.built.get(d) {
                if a.checkpoints == cps {
                    return *a;
                }
            }
            let base = key_of(&name);
            let mut key = base.clone();
            let mut n = 2;
            while reg.built.iter().any(|(p, a)| a.key == key && p != d) {
                key = format!("{base}_{n}");
                n += 1;
            }
            let app: &'static CustomApp = Box::leak(Box::new(CustomApp {
                subdir: leak(format!("custom/{key}")),
                key: leak(key),
                label: leak(format!("{name} (your TotalSynth checkpoints)")),
                folder: d.clone(),
                checkpoints: cps,
            }));
            reg.built.insert(d.clone(), app);
            app
        });
        match res {
            Ok(a) => {
                if !reg.live.iter().any(|l| l.key == a.key) {
                    reg.live.push(a);
                }
                out.push((d.clone(), Ok(a.key)));
            }
            Err(e) => out.push((d.clone(), Err(e))),
        }
    }
    out
}

/// The registered folders' apps.
pub fn apps() -> Vec<App> {
    REGISTRY
        .read()
        .ok()
        .and_then(|r| {
            r.as_ref()
                .map(|r| r.live.iter().map(|a| App::Custom(a)).collect())
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/synth-unetpp-cv0.pt")
    }

    #[test]
    fn a_folder_of_checkpoints_is_one_app() {
        let dir = std::env::temp_dir().join(format!("rds_synth_custom_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let d = dir.join("Pelvis MR-Linac");
        std::fs::create_dir_all(&d).unwrap();
        for name in ["CV_1.pt", "CV_0.pt", "extra.pt"] {
            std::fs::copy(fixture(), d.join(name)).unwrap();
        }
        std::fs::write(d.join("notes.txt"), "not a checkpoint").unwrap();
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        let bad = dir.join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("CV_0.pt"), b"no").unwrap();

        let res = register(&[d.clone(), empty.clone(), bad.clone()]);
        assert_eq!(res.len(), 3);
        let key = *res[0].1.as_ref().unwrap();
        assert_eq!(key, "sct_custom_pelvis_mr_linac");
        assert!(res[1].1.is_err() && res[2].1.is_err());
        let apps = apps();
        assert_eq!(apps.len(), 1);
        let App::Custom(a) = apps[0] else {
            panic!("not a custom app")
        };
        let names: Vec<String> = a
            .checkpoints
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, ["CV_0.pt", "CV_1.pt", "extra.pt"]);
        assert_eq!(apps[0].checkpoint_count(), 3);
        assert_eq!(App::from_key(key), Some(apps[0]));
        // Registering again hands out the same row; an empty list clears.
        let again = register(std::slice::from_ref(&d));
        assert_eq!(*again[0].1.as_ref().unwrap(), key);
        register(&[]);
        assert!(super::apps().is_empty());
        assert_eq!(App::from_key(key), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
