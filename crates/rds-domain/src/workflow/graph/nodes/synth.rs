//! The synthetic CT steps: making one ([`crate::sct::synthesize`]) and
//! judging one ([`crate::sct::eval::evaluate`]).
//!
//! *Synthetic CT* is the first step whose output is an image: the series it
//! writes joins the study (`Dataset::touched_series` records it, so *Export
//! DICOM* writes it with what the run made), its voxels go into the run's
//! volume cache, and the image wire it puts out is that series, which every
//! later step takes like any other.

use anyhow::{anyhow, bail, Result};

use super::super::catalog as cat;
use super::super::exec::{Ctx, Done, RegEntry, Report, Table, Value};
use super::super::Node;
use super::{count, done, image_of, r1, r2};
use crate::bodymask;
use crate::dicom_export;
use crate::loader;
use crate::models;
use crate::progress::Progress;
use crate::registration::{resample_through, Direction};
use crate::sct::{self, eval, BodySource, SctRequest};
use crate::synth::{self, App};
use crate::workflow::select;

/// A structure by name on one image as a mask on `grid`.
fn named_mask(
    ctx: &Ctx,
    ds: usize,
    uid: &str,
    name: &str,
    grid: &crate::volume::Grid,
) -> Result<Vec<u8>> {
    let study = &ctx.ds(ds)?.study;
    let s = select::find_on_series(study, name, uid, &grid.frame_of_reference_uid)
        .ok_or_else(|| anyhow!("no structure '{name}' on the image"))?;
    s.mask_on(grid)
}

pub(in super::super) fn synthesize_ct(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::SynthesizeCt,
    p: &Progress,
) -> Result<Done> {
    let (ds, uid) = image_of(v)?;
    let source = ctx
        .ds(ds)?
        .study
        .series
        .iter()
        .find(|s| s.uid == uid)
        .cloned()
        .ok_or_else(|| anyhow!("the image series is no longer in the study"))?;
    let app = App::from_key(&prm.app)
        .ok_or_else(|| anyhow!("'{}' is not a synthesis model", prm.app.trim()))?;
    if !app.modality().accepts(&source.modality) {
        bail!(
            "{} turns {} into a CT; the image is {}",
            app.label(),
            app.modality().label(),
            source.modality
        );
    }
    let checkpoints: Vec<usize> = if prm.all_checkpoints {
        (0..app.checkpoint_count()).collect()
    } else {
        vec![0]
    };
    let root = ctx.opts.models_dir.clone();
    let mut need = app.download_needed(&checkpoints, &root);
    let body_key = BodySource::from_key(&prm.mask);
    if body_key == Some(BodySource::Network) {
        let model = if source.modality.eq_ignore_ascii_case("MR") {
            bodymask::BodyModel::Mr6mm
        } else {
            bodymask::BodyModel::Ct6mm
        };
        need += bodymask::download_needed(model, &root);
    }
    if need > 0 && !ctx.opts.allow_download {
        bail!(
            "the {} weights are not present ({} to download) and downloads are off for this run",
            app.label(),
            models::human_bytes(need)
        );
    }
    let volume = ctx.volume(ds, &uid, p)?;
    let body = match body_key {
        Some(b) => b,
        None => BodySource::Structure {
            name: prm.mask.trim().to_string(),
            mask: named_mask(ctx, ds, &uid, prm.mask.trim(), &volume.grid())?,
        },
    };
    let req = SctRequest {
        app,
        opts: synth::Options {
            checkpoints,
            tta: prm.tta.min(synth::TTA_MAX),
            uncertainty: prm.uncertainty,
            device: prm.device.pref(),
        },
        body,
        series_number: sct::next_series_number(&ctx.ds(ds)?.study),
    };
    let label = ctx.ds(ds)?.label.clone();
    let out_root = ctx.out_path(&prm.folder, &label);
    let o = sct::synthesize(&volume, &source, &req, &root, &out_root, p)?;
    let new_uid = o.series.series_uid.clone();
    let mut made = vec![new_uid.clone()];
    if let Some((_, s)) = &o.uncertainty {
        made.push(s.series_uid.clone());
    }
    {
        let d = ctx.ds_mut(ds)?;
        let notes = loader::merge_study(&mut d.study, o.study.clone());
        if !notes.is_empty() {
            ctx.channel.log(notes.join("; "));
        }
        let d = ctx.ds_mut(ds)?;
        d.touched_series.extend(made.iter().cloned());
    }
    if let Some(info) = ctx
        .ds(ds)?
        .study
        .series
        .iter()
        .find(|s| s.uid == new_uid)
        .cloned()
    {
        ctx.volumes.put(&info, o.volume.clone());
    }
    let mut table = Table::new(
        "Synthetic CT",
        &[
            "Series",
            "Model",
            "Checkpoints",
            "Mirrors",
            "Members",
            "Air outside",
            "Device",
            "Time (s)",
        ],
    );
    table.row(vec![
        o.description.clone(),
        o.app_label.to_string(),
        o.checkpoints
            .iter()
            .map(|k| format!("CV_{k}"))
            .collect::<Vec<_>>()
            .join(" "),
        o.tta.to_string(),
        o.members.to_string(),
        o.body.clone(),
        o.device.clone(),
        r1(o.elapsed_secs),
    ]);
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Synthetic CT: {}", node.label()),
        notes: vec![format!(
            "Weights: {} ({}). Written to {}.",
            app.label(),
            app.licence().name(),
            out_root.display()
        )],
        tables: vec![table],
        motion: None,
    });
    let mut d = done(
        vec![
            Value::Image {
                ds,
                uid: new_uid.clone(),
            },
            Value::Report(report),
        ],
        vec![
            o.description.clone(),
            format!(
                "{} on {} in {} s",
                count(o.members, "member", "members"),
                if o.device.is_empty() {
                    "CPU"
                } else {
                    &o.device
                },
                r1(o.elapsed_secs)
            ),
        ],
    );
    d.touched.push((ds, Some(new_uid)));
    d.report = Some(report);
    Ok(d)
}

// The step's four inputs, its parameters and the run's state.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn evaluate_sct(
    ctx: &mut Ctx,
    node: &Node,
    sct_v: &Value,
    ref_v: &Value,
    reg_v: Option<&Value>,
    mask_v: Option<&Value>,
    prm: &cat::EvaluateSct,
    p: &Progress,
) -> Result<Done> {
    let (ds, uid) = image_of(sct_v)?;
    let (rds, ruid) = image_of(ref_v)?;
    if ds == rds && uid == ruid {
        bail!("the synthetic CT and the reference are the same series");
    }
    let align = match reg_v {
        Some(Value::Registration(i)) => match ctx.regs.get(*i) {
            Some(RegEntry::Pair {
                fixed,
                moving,
                result,
            }) => {
                let me = (ds, uid.clone());
                let other = (rds, ruid.clone());
                let dir = if *fixed == me && *moving == other {
                    Direction::FixedToMoving
                } else if *moving == me && *fixed == other {
                    Direction::MovingToFixed
                } else {
                    bail!("the registration is not between these two images");
                };
                Some((result.transform.clone(), dir))
            }
            _ => bail!("the registration must be one of two images, not of a 4D group"),
        },
        Some(_) => bail!("the Registration input needs a registration"),
        None => None,
    };
    let root = ctx.opts.models_dir.clone();
    if prm.structures {
        let need = crate::zoo::AutoModel::Nn(&crate::autoseg::total::TOTAL_FAST)
            .download_needed(None, &root);
        if need > 0 && !ctx.opts.allow_download {
            bail!(
                "the TotalSegmentator weights are not present ({} to download) and downloads \
                 are off for this run",
                models::human_bytes(need)
            );
        }
    }
    let sct_vol = ctx.volume(ds, &uid, p)?;
    let ref_vol = ctx.volume(rds, &ruid, p)?;
    if align.is_none() && sct_vol.frame_of_reference_uid != ref_vol.frame_of_reference_uid {
        bail!(
            "the two images are on different frames of reference: wire a registration of them \
             into Registration (Register, or Register by structures across MR and CT)"
        );
    }
    let grid = sct_vol.grid();
    p.set("Resampling the reference onto the synthetic CT");
    let reference = resample_through(
        &ref_vol,
        &grid,
        align.as_ref().map(|(t, d)| (t.as_ref(), *d)),
    );
    let (mask, mask_name) = match mask_v {
        Some(Value::Structures { ds: mds, names, .. }) => {
            if *mds != ds {
                bail!("the mask must be structures of the synthetic CT's study");
            }
            let mut acc = vec![0u8; sct_vol.data.len()];
            for n in names {
                let m = named_mask(ctx, ds, &uid, n, &grid)?;
                for (a, b) in acc.iter_mut().zip(m) {
                    *a |= b;
                }
            }
            (Some(acc), names.join(" + "))
        }
        Some(_) => bail!("the Mask input needs structures"),
        None => {
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
            let b = bodymask::contour_body(
                &sct::on_grid(&sct_vol, ct),
                &bodymask::BodyParams {
                    make_external: false,
                    ..bodymask::BodyParams::for_modality("CT")
                },
                &root,
                p,
            )?;
            (Some(b.mask), "the reference's body".to_string())
        }
    };
    let opts = eval::EvalOptions {
        wet: if prm.wet {
            Some(eval::Hlut::parse(&prm.hlut)?)
        } else {
            None
        },
        structures: prm.structures,
        device: prm.device.pref(),
    };
    let r = eval::evaluate(&sct_vol, &reference, mask.as_deref(), &opts, &root, p)?;

    let mut hu = Table::new(
        "HU",
        &["Voxels", "Class", "MAE (HU)", "ME (HU)", "RMSE (HU)"],
    );
    let mut row = |name: &str, m: &eval::Metrics| {
        hu.row(vec![
            m.voxels.to_string(),
            name.to_string(),
            r1(m.mae),
            r1(m.me),
            r1(m.rmse),
        ])
    };
    row("all", &r.overall);
    for c in &r.classes {
        row(c.name, &c.m);
    }
    let mut image = Table::new("Image", &["PSNR (dB)", "SSIM"]);
    image.row(vec![r2(r.psnr_db), format!("{:.4}", r.ssim)]);
    let mut tables = vec![hu, image];
    if !r.wet.is_empty() {
        let mut t = Table::new(
            "Water-equivalent thickness",
            &[
                "Axis",
                "Rays",
                "Mean WET (mm)",
                "Mean diff (mm)",
                "Mean |diff| (mm)",
                "95% |diff| (mm)",
                "Mean |diff| (%)",
            ],
        );
        for w in &r.wet {
            t.row(vec![
                w.axis.clone(),
                w.rays.to_string(),
                r1(w.mean_wet_mm),
                r2(w.mean_diff_mm),
                r2(w.mean_abs_diff_mm),
                r2(w.p95_abs_diff_mm),
                r2(w.mean_abs_rel_pct),
            ]);
        }
        tables.push(t);
    }
    if !r.dice.is_empty() {
        let mut t = Table::new(
            "Structures",
            &["Structure", "Dice", "CT (cm3)", "sCT (cm3)"],
        );
        for d in &r.dice {
            t.row(vec![
                d.name.clone(),
                format!("{:.3}", d.dice),
                r1(d.cm3_ct),
                r1(d.cm3_sct),
            ]);
        }
        tables.push(t);
    }
    let mut notes = vec![format!(
        "Inside {mask_name}; the reference {}.",
        if align.is_some() {
            "through the registration"
        } else {
            "on the same frame of reference"
        }
    )];
    let mut touched = Vec::new();
    if prm.difference {
        let label = ctx.ds(ds)?.label.clone();
        let out_root = ctx.out_path(&prm.folder, &label);
        let sct_series = ctx
            .ds(ds)?
            .study
            .series
            .iter()
            .find(|s| s.uid == uid)
            .cloned()
            .ok_or_else(|| anyhow!("the synthetic CT is no longer in the study"))?;
        let vol = sct::on_grid(&sct_vol, r.difference.clone());
        let duid = dicom_export::new_uid();
        let desc = sct::derived_description("minus CT", &sct_series.description);
        let written = dicom_export::write_derived_image_series(
            &vol,
            &sct_series,
            &sct::series_dir(&out_root, &sct_series.study_uid, &duid),
            &dicom_export::DerivedImage {
                series_uid: &duid,
                description: &desc,
                derivation: "Synthetic CT minus the reference CT, HU, on the synthetic CT's grid; \
                             0 where the reference has no value.",
                series_number: sct::next_series_number(&ctx.ds(ds)?.study),
                window: (0.0, 400.0),
                made_by: "Synthetic CT evaluation",
            },
            p,
        )?;
        let st = loader::load_files_to_merge(&written.files, "difference", p)?;
        let d = ctx.ds_mut(ds)?;
        loader::merge_study(&mut d.study, st);
        d.touched_series.insert(duid.clone());
        notes.push(format!("The difference series: {desc}."));
        touched.push((ds, Some(duid)));
    }
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Synthetic CT evaluation: {}", node.label()),
        notes,
        tables,
        motion: None,
    });
    let mut lines = vec![format!(
        "MAE {} HU, ME {} HU over {} voxels",
        r1(r.overall.mae),
        r1(r.overall.me),
        r.overall.voxels
    )];
    lines.push(format!("PSNR {} dB, SSIM {:.3}", r2(r.psnr_db), r.ssim));
    if let Some(m) = r.mean_dice {
        lines.push(format!(
            "mean Dice {m:.3} over {}",
            count(r.dice.len(), "structure", "structures")
        ));
    }
    let mut d = done(vec![Value::Report(report)], lines);
    d.touched = touched;
    d.report = Some(report);
    Ok(d)
}
