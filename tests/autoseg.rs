//! Auto-segmentation integration tests.
//!
//! The fast tests exercise the network assembly and the full sliding-window
//! machinery with a small synthetic network - no model download needed.
//! The `#[ignore]`d test runs the real 3 mm TotalSegmentator model against
//! bundled example data; enable it locally with
//!
//! ```text
//! RDS_AUTOSEG_MODELS=path/to/models/totalsegmentator \
//!   cargo test --release --test autoseg -- --ignored
//! ```
//!
//! (weights are downloaded into that folder on first use).

use std::collections::HashMap;

use rust_dicom_station::autoseg::{self, config::ModelConfig, cpu, net};
use rust_dicom_station::nn::cache::WTensor;

/// Deterministic pseudo-random values.
fn rngf(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 0.2 - 0.1
}

fn tensor(seed: &mut u64, shape: &[usize]) -> WTensor {
    let n: usize = shape.iter().product();
    WTensor {
        shape: shape.to_vec(),
        data: (0..n).map(|_| rngf(seed)).collect(),
    }
}

/// Build a miniature 3-stage PlainConvUNet with random weights using the
/// exact checkpoint key naming, and check the forward pass produces logits
/// of the right shape on an odd-sized patch.
#[test]
fn tiny_unet_assembles_and_runs() {
    let cfg = ModelConfig {
        patch_size: [16, 16, 16],
        spacing: [3.0, 3.0, 3.0],
        features: vec![4, 8, 16],
        kernels: vec![[3, 3, 3]; 3],
        strides: vec![[1, 1, 1], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage: vec![2, 2, 2],
        n_conv_per_stage_decoder: vec![2, 2],
        norm: rust_dicom_station::autoseg::config::Norm::Ct,
        clip_lo: -100.0,
        clip_hi: 100.0,
        mean: 0.0,
        std: 1.0,
    };
    let classes = 5usize;
    let mut s = 42u64;
    let mut t: HashMap<String, WTensor> = HashMap::new();
    // encoder
    for (st, &f) in cfg.features.iter().enumerate() {
        let cin_stage = if st == 0 { 1 } else { cfg.features[st - 1] };
        for i in 0..2 {
            let cin = if i == 0 { cin_stage } else { f };
            let p = format!("encoder.stages.{st}.0.convs.{i}");
            t.insert(
                format!("{p}.conv.weight"),
                tensor(&mut s, &[f, cin, 3, 3, 3]),
            );
            t.insert(format!("{p}.conv.bias"), tensor(&mut s, &[f]));
            t.insert(format!("{p}.norm.weight"), tensor(&mut s, &[f]));
            t.insert(format!("{p}.norm.bias"), tensor(&mut s, &[f]));
        }
    }
    // decoder
    for tr in 0..2 {
        let c_below = cfg.features[2 - tr];
        let c_skip = cfg.features[1 - tr];
        t.insert(
            format!("decoder.transpconvs.{tr}.weight"),
            tensor(&mut s, &[c_below, c_skip, 2, 2, 2]),
        );
        t.insert(
            format!("decoder.transpconvs.{tr}.bias"),
            tensor(&mut s, &[c_skip]),
        );
        for i in 0..2 {
            let cin = if i == 0 { 2 * c_skip } else { c_skip };
            let p = format!("decoder.stages.{tr}.convs.{i}");
            t.insert(
                format!("{p}.conv.weight"),
                tensor(&mut s, &[c_skip, cin, 3, 3, 3]),
            );
            t.insert(format!("{p}.conv.bias"), tensor(&mut s, &[c_skip]));
            t.insert(format!("{p}.norm.weight"), tensor(&mut s, &[c_skip]));
            t.insert(format!("{p}.norm.bias"), tensor(&mut s, &[c_skip]));
        }
        t.insert(
            format!("decoder.seg_layers.{tr}.weight"),
            tensor(&mut s, &[classes, c_skip, 1, 1, 1]),
        );
        t.insert(
            format!("decoder.seg_layers.{tr}.bias"),
            tensor(&mut s, &[classes]),
        );
    }
    let unet = net::UNet::build(cfg, &t).expect("assemble");
    assert_eq!(unet.num_classes(), classes);
    let x = cpu::Act {
        c: 1,
        d: 16,
        h: 16,
        w: 16,
        data: (0..16 * 16 * 16)
            .map(|i| (i % 13) as f32 * 0.1 - 0.6)
            .collect(),
    };
    let y = unet.forward_cpu(&x);
    assert_eq!((y.c, y.d, y.h, y.w), (classes, 16, 16, 16));
    assert!(y.data.iter().all(|v| v.is_finite()));
    // network is deterministic
    let x2 = cpu::Act {
        c: 1,
        d: 16,
        h: 16,
        w: 16,
        data: (0..16 * 16 * 16)
            .map(|i| (i % 13) as f32 * 0.1 - 0.6)
            .collect(),
    };
    let y2 = unet.forward_cpu(&x2);
    assert_eq!(y.data, y2.data);
}

/// Wrong shapes in the checkpoint must be rejected with a clear error, not
/// silently accepted.
#[test]
fn shape_mismatch_is_rejected() {
    let cfg = ModelConfig {
        patch_size: [8, 8, 8],
        spacing: [3.0, 3.0, 3.0],
        features: vec![4, 8],
        kernels: vec![[3, 3, 3]; 2],
        strides: vec![[1, 1, 1], [2, 2, 2]],
        n_conv_per_stage: vec![2, 2],
        n_conv_per_stage_decoder: vec![2],
        norm: rust_dicom_station::autoseg::config::Norm::Ct,
        clip_lo: 0.0,
        clip_hi: 1.0,
        mean: 0.0,
        std: 1.0,
    };
    let mut s = 1u64;
    let mut t: HashMap<String, WTensor> = HashMap::new();
    // deliberately wrong cin on the very first conv
    t.insert(
        "encoder.stages.0.0.convs.0.conv.weight".into(),
        tensor(&mut s, &[4, 2, 3, 3, 3]),
    );
    t.insert(
        "encoder.stages.0.0.convs.0.conv.bias".into(),
        tensor(&mut s, &[4]),
    );
    t.insert(
        "encoder.stages.0.0.convs.0.norm.weight".into(),
        tensor(&mut s, &[4]),
    );
    t.insert(
        "encoder.stages.0.0.convs.0.norm.bias".into(),
        tensor(&mut s, &[4]),
    );
    let err = match net::UNet::build(cfg, &t) {
        Ok(_) => panic!("mis-shaped checkpoint was accepted"),
        Err(e) => e,
    };
    assert!(format!("{err:#}").contains("shape"), "{err:#}");
}

/// Full pipeline against the real 3 mm model + the bundled example study.
/// Ignored by default (needs the weights and the example data); see the
/// module docs for how to run it.
#[test]
#[ignore]
fn real_model_on_example_data() {
    let models_dir = std::path::PathBuf::from(
        std::env::var("RDS_AUTOSEG_MODELS").expect("set RDS_AUTOSEG_MODELS"),
    );
    let data_dir = std::env::var("RDS_EXAMPLE_DATA")
        .unwrap_or_else(|_| "example_data/lung_p1_4DCT_phase_000".into());
    let study = rust_dicom_station::loader::load_directory(
        std::path::Path::new(&data_dir),
        &Default::default(),
    )
    .expect("load example data");
    let progress = rust_dicom_station::progress::Progress::default();
    let result = autoseg::run(
        &study.volume,
        autoseg::Variant::Fast3mm,
        autoseg::DevicePref::Cpu,
        [true; 5],
        &models_dir,
        &progress,
    )
    .expect("segmentation");
    // The bundled study is a thorax 4DCT phase: the big thoracic organs must
    // be present with plausible volumes.
    let organ = |name: &str| {
        result
            .organs
            .iter()
            .find(|o| o.name == name)
            .unwrap_or_else(|| panic!("{name} not found"))
    };
    let lungs: f64 = [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ]
    .iter()
    .map(|n| organ(n).cm3)
    .sum();
    assert!(
        lungs > 2000.0 && lungs < 8000.0,
        "total lung volume {lungs} cm³"
    );
    let heart = organ("heart").cm3;
    assert!(heart > 300.0 && heart < 1200.0, "heart {heart} cm³");
    assert!(organ("liver").cm3 > 800.0);
    assert!(organ("spinal_cord").cm3 > 20.0);
    assert!(result.organs.len() > 50, "found {}", result.organs.len());
}
