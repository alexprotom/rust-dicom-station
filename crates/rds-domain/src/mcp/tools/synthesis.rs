//! Synthetic CT: making one from an MR or CBCT series, and judging one
//! against a CT of the same patient.

use anyhow::{anyhow, bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::Core;
use super::segment::{models_root, refuse_download};
use super::session::round2;
use crate::bodymask::{self, BodyModel};
use crate::loader;
use crate::progress::Progress;
use crate::registration::{resample_through, Direction};
use crate::sct::{self, eval, BodySource, SctRequest};
use crate::synth::{self, App};

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SynthArgs {
    /// Dataset handle.
    pub dataset: String,
    /// Series number (from describe_dataset) of the MR or CBCT; the
    /// displayed series when omitted.
    #[serde(default)]
    pub series: Option<u32>,
    /// `sct_mr` (MR of any sequence), `sct_cbct` (linac CBCT), `sct_mr_cbct`
    /// (one model for both), `sct_mr_dixon` (whole-body Dixon MR), or the
    /// key of a checkpoint folder added in the viewer's model manager.
    /// Default: `sct_mr` on MR, `sct_cbct` otherwise.
    #[serde(default)]
    pub app: Option<String>,
    /// Which cross-validation checkpoints to average (0 to 4). Empty: CV_0
    /// alone, the quick look.
    #[serde(default)]
    pub checkpoints: Vec<usize>,
    /// Average every checkpoint of the app (the published recipe);
    /// overrides `checkpoints`.
    #[serde(default)]
    pub all_checkpoints: bool,
    /// Test-time mirrors besides the plain pass: 0, 1 (top-bottom) or 2
    /// (and left-right, the published recipe).
    #[serde(default)]
    pub tta: usize,
    /// Also write the members' voxel-wise standard deviation (HU) as a
    /// series.
    #[serde(default)]
    pub uncertainty: bool,
    /// What sets the air outside the patient: `network` (the body contour
    /// guided by TotalSegmentator's 6 mm body network, upstream's choice),
    /// `classical` (threshold and morphology), `none`, or the name of a
    /// structure of the dataset (BODY, External).
    #[serde(default = "default_mask")]
    pub mask: String,
}

fn default_mask() -> String {
    "network".into()
}

/// The app a call names, or the one for the series' modality.
fn app_of(key: Option<&str>, modality: &str) -> Result<App> {
    match key {
        Some(k) => App::from_key(k.trim()).ok_or_else(|| {
            anyhow!(
                "unknown app '{k}'; the apps are {}",
                App::all()
                    .iter()
                    .map(|a| a.key())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }),
        None => Ok(if modality.eq_ignore_ascii_case("MR") {
            App::Mr
        } else {
            App::Cbct
        }),
    }
}

pub fn synthesize_ct(core: &mut Core, a: SynthArgs, p: &Progress) -> Result<Value> {
    let (series, source, next_number) = {
        let ds = core.session.dataset(&a.dataset)?;
        let series = core.session.series_index(ds, a.series)?;
        (
            series,
            ds.study.series[series].clone(),
            sct::next_series_number(&ds.study),
        )
    };
    let app = app_of(a.app.as_deref(), &source.modality)?;
    if !app.modality().accepts(&source.modality) {
        bail!(
            "{} turns {} into a CT; series {} is {}",
            app.label(),
            app.modality().label(),
            series + 1,
            clean_text(&source.modality)
        );
    }
    let checkpoints: Vec<usize> = if a.all_checkpoints {
        (0..app.checkpoint_count()).collect()
    } else if a.checkpoints.is_empty() {
        vec![0]
    } else {
        a.checkpoints.clone()
    };
    if let Some(k) = checkpoints.iter().find(|&&k| k >= app.checkpoint_count()) {
        bail!(
            "{} has {} checkpoints; there is no {k}",
            app.key(),
            app.checkpoint_count()
        );
    }
    if a.tta > synth::TTA_MAX {
        bail!("tta is 0, 1 or 2");
    }
    let root = models_root(core);
    refuse_download(core, app.download_needed(&checkpoints, &root), app.label())?;
    let volume = core.session.volume(&a.dataset, series, p)?;
    let body = match BodySource::from_key(&a.mask) {
        Some(BodySource::Network) => {
            let model = if source.modality.eq_ignore_ascii_case("MR") {
                BodyModel::Mr6mm
            } else {
                BodyModel::Ct6mm
            };
            refuse_download(
                core,
                bodymask::download_needed(model, &root),
                "body-outline",
            )?;
            BodySource::Network
        }
        Some(b) => b,
        None => {
            let s = core.session.structure(&a.dataset, &a.mask, None)?;
            BodySource::Structure {
                name: s.name.clone(),
                mask: s.mask_on(&volume.grid())?,
            }
        }
    };
    let req = SctRequest {
        app,
        opts: synth::Options {
            checkpoints,
            tta: a.tta,
            uncertainty: a.uncertainty,
            device: core.session.config.device_pref(),
        },
        body,
        series_number: next_number,
    };
    let out_root = core.session.out_dir()?.join("derived");
    let o = sct::synthesize(&volume, &source, &req, &root, &out_root, p)?;
    // The new series join the dataset; the sCT's voxels are kept so the
    // next tool does not read them back.
    let summary = sct::summary(&o);
    let ds = core.session.dataset_mut(&a.dataset)?;
    let notes = loader::merge_study(&mut ds.study, o.study.clone());
    let number_of = |uid: &str| {
        ds.study
            .series
            .iter()
            .position(|s| s.uid == uid)
            .map(|i| i + 1)
    };
    let sct_number = number_of(&o.series.series_uid);
    let sd_number = o
        .uncertainty
        .as_ref()
        .and_then(|(_, s)| number_of(&s.series_uid));
    if let Some(info) = ds
        .study
        .series
        .iter()
        .find(|s| s.uid == o.series.series_uid)
        .cloned()
    {
        core.session.volumes.put(&info, o.volume.clone());
    }
    Ok(json!({
        "dataset": a.dataset,
        "source_series": series + 1,
        "series": sct_number,
        "uncertainty_series": sd_number,
        "description": clean_text(&o.description),
        "app": app.key(),
        "app_label": app.label(),
        "licence": app.licence().name(),
        "checkpoints": o.checkpoints,
        "tta": o.tta,
        "members": o.members,
        "mask": o.body,
        "clip": [o.clip.0, round2(o.clip.1)],
        "working_grid": o.working_dims,
        "device": o.device,
        "elapsed_s": (o.elapsed_secs * 10.0).round() / 10.0,
        "summary": clean_text(&summary),
        "notes": notes,
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EvalArgs {
    /// Dataset of the synthetic CT.
    pub dataset: String,
    /// Series number of the synthetic CT; the displayed series when
    /// omitted.
    #[serde(default)]
    pub sct: Option<u32>,
    /// Dataset of the reference CT; the same dataset when omitted.
    #[serde(default)]
    pub reference_dataset: Option<String>,
    /// Series number of the reference CT in its dataset.
    pub reference: u32,
    /// A registration (from register or register_structures) between the
    /// synthetic CT and the reference, in either direction. Without one the
    /// two must share a frame of reference.
    #[serde(default)]
    pub reg: Option<String>,
    /// Where to compare: `body` (the reference's body contour), `covered`
    /// (everywhere the reference has a value), or a structure name of the
    /// synthetic CT's dataset.
    #[serde(default = "default_eval_mask")]
    pub mask: String,
    /// Water-equivalent thickness rows (protons), through `hlut`.
    #[serde(default)]
    pub wet: bool,
    /// HU to relative stopping power as `HU:RSP` pairs separated by commas;
    /// a generic curve when omitted (not a scanner's calibration).
    #[serde(default)]
    pub hlut: Option<String>,
    /// Run TotalSegmentator (total, 3 mm) on both and report the Dice of
    /// its structures (minutes; its weights must be present unless
    /// downloads are allowed).
    #[serde(default)]
    pub structures: bool,
}

fn default_eval_mask() -> String {
    "body".into()
}

pub fn evaluate_sct(core: &mut Core, a: EvalArgs, p: &Progress) -> Result<Value> {
    let ref_ds = a
        .reference_dataset
        .clone()
        .unwrap_or_else(|| a.dataset.clone());
    let sct_idx = {
        let ds = core.session.dataset(&a.dataset)?;
        core.session.series_index(ds, a.sct)?
    };
    let ref_idx = {
        let ds = core.session.dataset(&ref_ds)?;
        core.session.series_index(ds, Some(a.reference))?
    };
    if ref_ds == a.dataset && ref_idx == sct_idx {
        bail!("the synthetic CT and the reference are the same series");
    }
    let root = models_root(core);
    if a.structures {
        refuse_download(
            core,
            crate::zoo::AutoModel::Nn(&crate::autoseg::total::TOTAL_FAST)
                .download_needed(None, &root),
            "TotalSegmentator",
        )?;
    }
    let align = match &a.reg {
        Some(id) => {
            let r = core.session.registration(id)?;
            let me = (a.dataset.clone(), sct_idx);
            let other = (ref_ds.clone(), ref_idx);
            let dir = if r.fixed == me && r.moving == other {
                Direction::FixedToMoving
            } else if r.moving == me && r.fixed == other {
                Direction::MovingToFixed
            } else {
                bail!("registration {id} is not between these two series");
            };
            Some((r.result.transform.clone(), dir))
        }
        None => None,
    };
    let sct_vol = core.session.volume(&a.dataset, sct_idx, p)?;
    let ref_vol = core.session.volume(&ref_ds, ref_idx, p)?;
    if align.is_none() && sct_vol.frame_of_reference_uid != ref_vol.frame_of_reference_uid {
        bail!(
            "the two series are on different frames of reference; register them first \
             (register, or register_structures across MR and CT) and pass the reg handle"
        );
    }
    p.set("Resampling the reference onto the synthetic CT");
    let reference = resample_through(
        &ref_vol,
        &sct_vol.grid(),
        align.as_ref().map(|(t, d)| (t.as_ref(), *d)),
    );
    let (mask, mask_name) = match a.mask.trim().to_ascii_lowercase().as_str() {
        "covered" => (None, "covered".to_string()),
        "body" => {
            let ct: Vec<i16> = reference
                .iter()
                .map(|&v| {
                    if v.is_nan() {
                        -1024
                    } else {
                        v.round().clamp(-32768.0, 32767.0) as i16
                    }
                })
                .collect();
            let ct = sct::on_grid(&sct_vol, ct);
            let b = bodymask::contour_body(
                &ct,
                &bodymask::BodyParams {
                    make_external: false,
                    ..bodymask::BodyParams::for_modality("CT")
                },
                &root,
                p,
            )?;
            (Some(b.mask), "body".to_string())
        }
        _ => {
            let s = core.session.structure(&a.dataset, &a.mask, None)?;
            (Some(s.mask_on(&sct_vol.grid())?), s.name.clone())
        }
    };
    let wet = if a.wet {
        Some(match &a.hlut {
            Some(t) => eval::Hlut::parse(t)?,
            None => eval::Hlut::default(),
        })
    } else {
        None
    };
    let opts = eval::EvalOptions {
        wet,
        structures: a.structures,
        device: core.session.config.device_pref(),
    };
    let r = eval::evaluate(&sct_vol, &reference, mask.as_deref(), &opts, &root, p)?;
    let metrics = |m: &eval::Metrics| {
        json!({
            "voxels": m.voxels,
            "mae_hu": round2(m.mae),
            "me_hu": round2(m.me),
            "rmse_hu": round2(m.rmse),
        })
    };
    let classes: Vec<Value> = r
        .classes
        .iter()
        .map(|c| {
            let mut v = metrics(&c.m);
            v["class"] = json!(c.name);
            v
        })
        .collect();
    let wet_rows: Vec<Value> = r
        .wet
        .iter()
        .map(|w| {
            json!({
                "axis": w.axis,
                "rays": w.rays,
                "mean_wet_mm": round2(w.mean_wet_mm),
                "mean_diff_mm": round2(w.mean_diff_mm),
                "mean_abs_diff_mm": round2(w.mean_abs_diff_mm),
                "p95_abs_diff_mm": round2(w.p95_abs_diff_mm),
                "mean_abs_rel_pct": round2(w.mean_abs_rel_pct),
            })
        })
        .collect();
    let dice: Vec<Value> = r
        .dice
        .iter()
        .map(|d| json!({"structure": d.name, "dice": (d.dice * 1000.0).round() / 1000.0}))
        .collect();
    Ok(json!({
        "dataset": a.dataset,
        "sct_series": sct_idx + 1,
        "reference_dataset": ref_ds,
        "reference_series": ref_idx + 1,
        "aligned_by": a.reg.clone().unwrap_or_else(|| "same frame of reference".into()),
        "mask": clean_text(&mask_name),
        "overall": metrics(&r.overall),
        "classes": classes,
        "psnr_db": round2(r.psnr_db),
        "ssim": (r.ssim * 10000.0).round() / 10000.0,
        "wet": wet_rows,
        "dice": dice,
        "mean_dice": r.mean_dice.map(|m| (m * 1000.0).round() / 1000.0),
        "csv": eval::csv(&r),
        "elapsed_s": (r.elapsed_secs * 10.0).round() / 10.0,
    }))
}
