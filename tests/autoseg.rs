//! Auto-segmentation integration tests.
//!
//! The fast tests exercise the network assembly and the full sliding-window
//! machinery with a small synthetic network - no model download needed.
//! The `#[ignore]`d test runs the real 3 mm TotalSegmentator model against
//! bundled example data; enable it locally with
//!
//! ```text
//! RDS_AUTOSEG_MODELS=path/to/models \
//!   cargo test --release --test autoseg -- --ignored
//! ```
//!
//! (weights are downloaded into that folder on first use).

use std::collections::HashMap;

use rust_dicom_station::autoseg::{self, config::Arch, config::ModelConfig, cpu, net};
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
        arch: Arch::PlainConv,
        conv_bias: true,
        patch_size: [16, 16, 16],
        spacing: [3.0, 3.0, 3.0],
        features: vec![4, 8, 16],
        kernels: vec![[3, 3, 3]; 3],
        strides: vec![[1, 1, 1], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage: vec![2, 2, 2],
        n_conv_per_stage_decoder: vec![2, 2],
        decoder_kernels: vec![],
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

/// Build a miniature 3-stage ResidualEncoderUNet with random weights using
/// the exact checkpoint key naming of the v3 `small` models (stem,
/// `stages.{s}.blocks.{b}.{conv1,conv2}`, a `skip.1` conv+norm where the
/// width changes and none where only the stride does), and check the
/// forward pass produces logits of the right shape.
#[test]
fn tiny_residual_unet_assembles_and_runs() {
    // Stage 2 keeps stage 1's width with a stride, like v3 small's last
    // stage: its skip path is the average pool alone, no tensors.
    let features = vec![4usize, 8, 8];
    let blocks_per_stage = vec![1usize, 2, 2];
    let cfg = ModelConfig {
        arch: Arch::ResidualEncoder {
            blocks_per_stage: blocks_per_stage.clone(),
        },
        conv_bias: true,
        patch_size: [16, 16, 16],
        spacing: [3.0, 3.0, 3.0],
        features: features.clone(),
        kernels: vec![[3, 3, 3]; 3],
        strides: vec![[1, 1, 1], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage: vec![],
        n_conv_per_stage_decoder: vec![1, 1],
        decoder_kernels: vec![],
        norm: rust_dicom_station::autoseg::config::Norm::Ct,
        clip_lo: -100.0,
        clip_hi: 100.0,
        mean: 0.0,
        std: 1.0,
    };
    let classes = 3usize;
    let mut s = 7u64;
    let mut t: HashMap<String, WTensor> = HashMap::new();
    let conv_norm = |t: &mut HashMap<String, WTensor>,
                     s: &mut u64,
                     p: &str,
                     cout: usize,
                     cin: usize,
                     k: usize| {
        t.insert(format!("{p}.conv.weight"), tensor(s, &[cout, cin, k, k, k]));
        t.insert(format!("{p}.conv.bias"), tensor(s, &[cout]));
        t.insert(format!("{p}.norm.weight"), tensor(s, &[cout]));
        t.insert(format!("{p}.norm.bias"), tensor(s, &[cout]));
    };
    conv_norm(&mut t, &mut s, "encoder.stem.convs.0", features[0], 1, 3);
    for (st, &f) in features.iter().enumerate() {
        let cin_stage = if st == 0 {
            features[0]
        } else {
            features[st - 1]
        };
        for b in 0..blocks_per_stage[st] {
            let cin = if b == 0 { cin_stage } else { f };
            let p = format!("encoder.stages.{st}.blocks.{b}");
            conv_norm(&mut t, &mut s, &format!("{p}.conv1"), f, cin, 3);
            conv_norm(&mut t, &mut s, &format!("{p}.conv2"), f, f, 3);
            if cin != f {
                t.insert(
                    format!("{p}.skip.1.conv.weight"),
                    tensor(&mut s, &[f, cin, 1, 1, 1]),
                );
                t.insert(format!("{p}.skip.1.norm.weight"), tensor(&mut s, &[f]));
                t.insert(format!("{p}.skip.1.norm.bias"), tensor(&mut s, &[f]));
            }
        }
    }
    for tr in 0..2 {
        let c_below = features[2 - tr];
        let c_skip = features[1 - tr];
        t.insert(
            format!("decoder.transpconvs.{tr}.weight"),
            tensor(&mut s, &[c_below, c_skip, 2, 2, 2]),
        );
        t.insert(
            format!("decoder.transpconvs.{tr}.bias"),
            tensor(&mut s, &[c_skip]),
        );
        conv_norm(
            &mut t,
            &mut s,
            &format!("decoder.stages.{tr}.convs.0"),
            c_skip,
            2 * c_skip,
            3,
        );
        t.insert(
            format!("decoder.seg_layers.{tr}.weight"),
            tensor(&mut s, &[classes, c_skip, 1, 1, 1]),
        );
        t.insert(
            format!("decoder.seg_layers.{tr}.bias"),
            tensor(&mut s, &[classes]),
        );
    }
    let n_tensors = t.len();
    let unet = net::UNet::build(cfg.clone(), &t).expect("assemble residual network");
    assert_eq!(unet.num_classes(), classes);
    assert!(unet.stem.is_some());
    let x = cpu::Act {
        c: 1,
        d: 16,
        h: 16,
        w: 16,
        data: (0..16 * 16 * 16)
            .map(|i| (i % 11) as f32 * 0.1 - 0.5)
            .collect(),
    };
    let y = unet.forward_cpu(&x);
    assert_eq!((y.c, y.d, y.h, y.w), (classes, 16, 16, 16));
    assert!(y.data.iter().all(|v| v.is_finite()));
    // A projection tensor that is missing is reported by name, and one that
    // is present where the width does not change is not read at all.
    let mut missing = t.clone();
    missing.remove("encoder.stages.1.blocks.0.skip.1.conv.weight");
    let err = net::UNet::build(cfg.clone(), &missing)
        .err()
        .expect("must fail");
    assert!(
        format!("{err:#}").contains("encoder.stages.1.blocks.0.skip.1.conv.weight"),
        "{err:#}"
    );
    let mut extra = t;
    extra.insert(
        "encoder.stages.2.blocks.0.skip.1.conv.weight".into(),
        tensor(&mut s, &[8, 8, 1, 1, 1]),
    );
    assert_eq!(extra.len(), n_tensors + 1);
    assert!(net::UNet::build(cfg, &extra).is_ok());
}

/// Wrong shapes in the checkpoint must be rejected with a clear error, not
/// silently accepted.
#[test]
fn shape_mismatch_is_rejected() {
    let cfg = ModelConfig {
        arch: Arch::PlainConv,
        conv_bias: true,
        patch_size: [8, 8, 8],
        spacing: [3.0, 3.0, 3.0],
        features: vec![4, 8],
        kernels: vec![[3, 3, 3]; 2],
        strides: vec![[1, 1, 1], [2, 2, 2]],
        n_conv_per_stage: vec![2, 2],
        n_conv_per_stage_decoder: vec![2],
        decoder_kernels: vec![],
        norm: rust_dicom_station::autoseg::config::Norm::Ct,
        clip_lo: 0.0,
        clip_hi: 1.0,
        mean: 0.0,
        std: 1.0,
    };
    let mut s = 1u64;
    let mut t: HashMap<String, WTensor> = HashMap::new();
    // Deliberately wrong cout on the very first conv (its cin is read from
    // the checkpoint itself - nnInteractive's network takes eight - so a
    // wrong cin would simply be a different network).
    t.insert(
        "encoder.stages.0.0.convs.0.conv.weight".into(),
        tensor(&mut s, &[5, 1, 3, 3, 3]),
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
fn real_model_on_test_data() {
    let models_dir = std::path::PathBuf::from(
        std::env::var("RDS_AUTOSEG_MODELS").expect("set RDS_AUTOSEG_MODELS"),
    );
    let data_dir = std::env::var("RDS_EXAMPLE_DATA").unwrap_or_else(|_| {
        "data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS/1_CT_4DFBCT__Gated__0.0_A".into()
    });
    let study = rust_dicom_station::loader::load_directory(
        std::path::Path::new(&data_dir),
        &Default::default(),
    )
    .expect("load example data");
    let progress = rust_dicom_station::progress::Progress::default();
    let result = autoseg::run(
        &study.volume,
        autoseg::Variant::Fast3mm.task(),
        &autoseg::NnOptions {
            device: autoseg::DevicePref::Cpu,
            parts: None,
        },
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
