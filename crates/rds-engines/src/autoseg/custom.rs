//! Model folders the user adds: an nnU-Net v2 model trained elsewhere, run
//! like any built-in task.
//!
//! What is accepted is one training folder as nnU-Net v2 writes it
//! (`<trainer>__<plans>__<configuration>/` with `plans.json`,
//! `dataset.json` and `fold_<k>/checkpoint_final.pth`), or the dataset
//! folder above it when it holds exactly one training. The network must be
//! one this engine assembles (a 3-D plain or residual-encoder U-Net) and
//! take one input channel; its classes are `dataset.json`'s labels. Every
//! fold found is ensembled. The image is read the way TotalSegmentator's
//! models are trained, reoriented to [S, A, R]; a model trained on other
//! orientations relies on its mirroring augmentation.
//!
//! The folder is only read: the converted weights go to the model folder's
//! `custom/<key>/`. Registered tasks live for the rest of the process (the
//! registry hands out `'static` rows, as the built-in ones are).

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use super::config::ModelConfig;
use super::task::{FoldUse, Licence, Modality, NnTask, Part, Post, IDENTITY_LUT};
use super::weights::{Home, ModelSpec, Source, PLANS_NAME};
use crate::volume::AxisOrder;

/// What a model folder holds.
#[derive(Clone, Debug, PartialEq)]
pub struct FolderInfo {
    /// The training folder (the one with `plans.json`).
    pub training: PathBuf,
    /// `Dataset123_Name` (or the training folder's own name).
    pub name: String,
    pub configuration: String,
    pub modality: Modality,
    /// Label `l` is `classes[l - 1]`.
    pub classes: Vec<String>,
    pub folds: Vec<u8>,
}

/// The training folder in `dir`: `dir` itself, or its one sub-folder with
/// a `plans.json`.
fn training_folder(dir: &Path) -> Result<PathBuf> {
    if dir.join(PLANS_NAME).is_file() {
        return Ok(dir.to_path_buf());
    }
    let subs: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join(PLANS_NAME).is_file())
        .collect();
    match subs.len() {
        1 => Ok(subs.into_iter().next().unwrap()),
        0 => bail!(
            "{} holds no nnU-Net training (no plans.json)",
            dir.display()
        ),
        n => bail!(
            "{} holds {n} trainings; choose one of its sub-folders",
            dir.display()
        ),
    }
}

/// Read and check a model folder.
pub fn describe(dir: &Path) -> Result<FolderInfo> {
    let training = training_folder(dir)?;
    let leaf = training
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let configuration = leaf
        .rsplit("__")
        .next()
        .filter(|c| c.starts_with("3d_") || c.starts_with("2d"))
        .unwrap_or("3d_fullres")
        .to_string();
    let plans = std::fs::read_to_string(training.join(PLANS_NAME))
        .with_context(|| format!("read {}", training.join(PLANS_NAME).display()))?;
    ModelConfig::from_plans_json_cfg(&plans, &configuration)
        .with_context(|| format!("{}: configuration {configuration}", training.display()))?;
    let ds_path = ["dataset.json", "../dataset.json"]
        .iter()
        .map(|n| training.join(n))
        .find(|p| p.is_file())
        .with_context(|| format!("{}: no dataset.json", training.display()))?;
    let ds: Value = serde_json::from_str(
        &std::fs::read_to_string(&ds_path)
            .with_context(|| format!("read {}", ds_path.display()))?,
    )
    .with_context(|| format!("parse {}", ds_path.display()))?;
    let channels = ds["channel_names"]
        .as_object()
        .or_else(|| ds["modality"].as_object())
        .context("dataset.json: no channel_names")?;
    if channels.len() != 1 {
        bail!(
            "dataset.json: {} input channels; only one-channel models run here",
            channels.len()
        );
    }
    let ch = channels
        .values()
        .next()
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let modality = if ch.eq_ignore_ascii_case("CT") {
        Modality::Ct
    } else {
        Modality::Mr
    };
    let labels = ds["labels"]
        .as_object()
        .context("dataset.json: no labels")?;
    let mut by_value: Vec<(u64, String)> = Vec::new();
    for (name, v) in labels {
        let Some(l) = v.as_u64() else {
            bail!(
                "dataset.json: label {name:?} is a region; region-based models are not supported"
            );
        };
        if l > 0 {
            by_value.push((l, name.clone()));
        }
    }
    by_value.sort();
    let max = by_value.last().map(|(l, _)| *l).unwrap_or(0);
    if max == 0 || max > 255 {
        bail!("dataset.json: labels must run from 1 to at most 255");
    }
    let classes: Vec<String> = (1..=max)
        .map(|l| {
            by_value
                .iter()
                .find(|(v, _)| *v == l)
                .map(|(_, n)| n.clone())
                .unwrap_or_else(|| format!("label_{l}"))
        })
        .collect();
    let mut folds: Vec<u8> = (0..10u8)
        .filter(|f| {
            let d = training.join(format!("fold_{f}"));
            d.join("checkpoint_final.pth").is_file() || d.join("checkpoint_best.pth").is_file()
        })
        .collect();
    folds.sort();
    if folds.is_empty() {
        bail!("{}: no fold_<k>/checkpoint_final.pth", training.display());
    }
    if folds != (0..folds.len() as u8).collect::<Vec<_>>() {
        bail!(
            "{}: the folds must be numbered from 0 without gaps (found {folds:?})",
            training.display()
        );
    }
    let name = training
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| n.starts_with("Dataset"))
        .unwrap_or(leaf);
    Ok(FolderInfo {
        training,
        name,
        configuration,
        modality,
        classes,
        folds,
    })
}

/// The registry key of a folder: `custom_` and its name in lower case.
pub fn key_of(info: &FolderInfo) -> String {
    let mut k = String::from("custom_");
    for c in info.name.chars() {
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

/// The task for a folder, built once per folder and process.
fn build(info: &FolderInfo, key: &str) -> &'static NnTask {
    let path = leak(info.training.to_string_lossy().to_string());
    let label = leak(info.name.clone());
    let classes: &'static [&'static str] = Box::leak(
        info.classes
            .iter()
            .map(|c| leak(c.clone()))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    );
    let spec = ModelSpec {
        key: leak(key.to_string()),
        label,
        detail: "An nnU-Net v2 model folder added on this computer.",
        url: path,
        zip_bytes: 0,
        folds: info.folds.len() as u8,
        axes: AxisOrder::Sar,
        home: Home::Custom,
        plans: "",
        configuration: leak(info.configuration.clone()),
        source: Source::Local,
    };
    let parts: &'static [Part] = Box::leak(Box::new([Part {
        spec,
        lut: &IDENTITY_LUT,
    }]));
    Box::leak(Box::new(NnTask {
        key: spec.key,
        label,
        group: "Your nnU-Net models",
        detail: spec.detail,
        modality: info.modality,
        licence: Licence::Undeclared,
        classes,
        parts,
        folds: FoldUse::All,
        step: 0.5,
        crop: None,
        post: Post::None,
    }))
}

struct Registry {
    /// Every folder ever built in this process, by path.
    built: HashMap<PathBuf, &'static NnTask>,
    /// The ones registered now, in order.
    live: Vec<&'static NnTask>,
}

static REGISTRY: RwLock<Option<Registry>> = RwLock::new(None);

/// Make `dirs` the registered model folders (replacing the previous set).
/// Returns, per folder, its task key or why it was refused.
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
    let builtin: Vec<&str> = super::task::builtin_tasks().iter().map(|t| t.key).collect();
    for d in dirs {
        let res = describe(d).map(|info| {
            if let Some(t) = reg.built.get(&info.training) {
                return *t;
            }
            let mut key = key_of(&info);
            let base = key.clone();
            let mut n = 2;
            while builtin.contains(&key.as_str()) || reg.built.values().any(|t| t.key == key) {
                key = format!("{base}_{n}");
                n += 1;
            }
            let t = build(&info, &key);
            reg.built.insert(info.training.clone(), t);
            t
        });
        match res {
            Ok(t) => {
                if !reg.live.iter().any(|l| l.key == t.key) {
                    reg.live.push(t);
                }
                out.push((d.clone(), Ok(t.key)));
            }
            Err(e) => out.push((d.clone(), Err(e))),
        }
    }
    out
}

/// The registered model folders' tasks.
pub fn tasks() -> Vec<&'static NnTask> {
    REGISTRY
        .read()
        .ok()
        .and_then(|r| r.as_ref().map(|r| r.live.clone()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A training folder with a plans.json from the bundled fixtures and a
    /// dataset.json; `folds` empty checkpoint files.
    fn folder(name: &str, labels: &str, folds: &[u8]) -> PathBuf {
        let root = std::env::temp_dir().join("rds_custom_models").join(name);
        let _ = std::fs::remove_dir_all(&root);
        let t = root.join("nnUNetTrainer__nnUNetPlans__3d_fullres");
        std::fs::create_dir_all(&t).unwrap();
        let plans = super::super::v1::plans_json(include_bytes!(
            "../../../../tests/data/nnunet-v1-lung-plans.pkl"
        ))
        .unwrap();
        std::fs::write(t.join("plans.json"), plans).unwrap();
        std::fs::write(
            t.join("dataset.json"),
            format!(r#"{{"channel_names": {{"0": "CT"}}, "labels": {labels}}}"#),
        )
        .unwrap();
        for f in folds {
            let d = t.join(format!("fold_{f}"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("checkpoint_final.pth"), b"").unwrap();
        }
        root
    }

    #[test]
    fn a_dataset_folder_with_one_training_is_read() {
        let root = folder(
            "Dataset042_Liver",
            r#"{"background": 0, "liver": 1, "tumour": 3}"#,
            &[0, 1],
        );
        let info = describe(&root).unwrap();
        assert_eq!(info.name, "Dataset042_Liver");
        assert_eq!(info.configuration, "3d_fullres");
        assert_eq!(info.modality, Modality::Ct);
        assert_eq!(info.classes, ["liver", "label_2", "tumour"]);
        assert_eq!(info.folds, [0, 1]);
        assert_eq!(key_of(&info), "custom_dataset042_liver");
        let reg = register(std::slice::from_ref(&root));
        let key = *reg[0].1.as_ref().unwrap();
        let t = tasks().into_iter().find(|t| t.key == key).unwrap();
        assert_eq!(t.classes.len(), 3);
        assert_eq!(t.parts[0].spec.folds, 2);
        assert_eq!(t.parts[0].spec.source, Source::Local);
        // Registering again hands out the same row.
        let again = register(&[root]);
        assert_eq!(*again[0].1.as_ref().unwrap(), key);
    }

    #[test]
    fn regions_and_missing_folds_are_refused() {
        let regions = folder(
            "Dataset043_Regions",
            r#"{"background": 0, "whole": [1, 2]}"#,
            &[0],
        );
        assert!(describe(&regions).is_err());
        let gap = folder("Dataset044_Gap", r#"{"background": 0, "a": 1}"#, &[1]);
        assert!(describe(&gap).is_err());
        let none = folder("Dataset045_None", r#"{"background": 0, "a": 1}"#, &[]);
        assert!(describe(&none).is_err());
    }
}
