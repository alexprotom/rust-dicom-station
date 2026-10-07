//! Development probe: dump one preprocessed patch and its logits so the
//! network can be verified numerically against PyTorch/nnU-Net.
//!
//! cargo run --release --example autoseg_probe -- <dicom_dir> <models_dir> <spec_key> <out_prefix>
//!
//! `<models_dir>` is the model folder; the spec's engine folder
//! (`totalsegmentator/`, `mrsegmentator/`) is below it.

use rust_dicom_station::autoseg::{config::ModelConfig, cpu, net, preprocess, weights};
use rust_dicom_station::loader;
use rust_dicom_station::progress::{Progress, Stderr};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut a = std::env::args().skip(1);
    let dicom = PathBuf::from(a.next().unwrap());
    let models = PathBuf::from(a.next().unwrap());
    let key = a.next().unwrap();
    let out = a.next().unwrap();
    let spec = weights::spec_by_key(&key).expect("unknown spec key");
    let study = loader::load_directory(&dicom, &Progress::default())?;
    let vol = &study.volume;
    let model = weights::ensure_model_folds(&spec, &models.join(spec.home.subdir()), 1, &Stderr)?;
    let mut cfg: ModelConfig = model.config.clone();

    let map = preprocess::SarMap::with_axes(vol, cfg.spacing, spec.axes);
    let vm = preprocess::resample_to_model(vol, &map);
    cfg.apply_image_norm(&vm);
    let unet = net::UNet::build(cfg.clone(), model.tensors())?;
    let cfg = &cfg;
    eprintln!(
        "model {} classes={} stages={}",
        spec.key,
        unet.num_classes(),
        cfg.n_stages()
    );
    eprintln!("model grid {:?}", map.model_dims);
    // extract the corner patch (like the first sliding-window tile)
    let [p0, p1, p2] = cfg.patch_size;
    let [d0, d1, d2] = map.model_dims;
    let inv_std = 1.0 / cfg.std.max(1e-8);
    let mut patch = vec![0f32; p0 * p1 * p2];
    for z in 0..p0 {
        for y in 0..p1 {
            for x in 0..p2 {
                let v = if z < d0 && y < d1 && x < d2 {
                    let raw = vm[(z * d1 + y) * d2 + x];
                    (raw.clamp(cfg.clip_lo, cfg.clip_hi) - cfg.mean) * inv_std
                } else {
                    0.0
                };
                patch[(z * p1 + y) * p2 + x] = v;
            }
        }
    }
    let x = cpu::Act {
        c: 1,
        d: p0,
        h: p1,
        w: p2,
        data: patch.clone(),
    };
    let t = std::time::Instant::now();
    let logits = unet.forward_cpu(&x);
    eprintln!(
        "forward {:.1}s -> [{} classes]",
        t.elapsed().as_secs_f64(),
        logits.c
    );
    let write = |name: &str, data: &[f32]| {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("{out}.{name}.bin"), bytes).unwrap();
    };
    write("patch", &patch);
    write("logits", &logits.data);
    eprintln!("wrote {out}.patch.bin / {out}.logits.bin");
    Ok(())
}
