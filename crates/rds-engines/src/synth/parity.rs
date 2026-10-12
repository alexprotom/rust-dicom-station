//! The engine against KonfAI itself (`tests/data/synth-*`).
//!
//! A miniature of the TotalSynth network (KonfAI's `UNetPlusPlus.yml`
//! with narrow stages, its batch norms calibrated on CT stacks, two seeds)
//! saved as KonfAI saves a checkpoint, with its output for one stack; and a
//! small CT block run through KonfAI's own `konfai PREDICTION` with the
//! ImpactSynth `Prediction.yml` (one checkpoint without mirrors, then both
//! with the two mirrors), with the body mask and the working-grid mask it
//! wrote. The network is matched to float rounding, the masks to the voxel
//! and the synthetic CT to the HU but for the float rounding: KonfAI keeps
//! the network's output in half precision, so the last bits PyTorch and
//! the GEMM here sum differently move a voxel by one half-precision step
//! at a few voxels in a thousand (each member the same, with the mirrors in
//! the right order).

use std::path::{Path, PathBuf};

use safetensors::{Dtype, SafeTensors};

use super::*;
use crate::geometry::Vec3;
use crate::nn::cache::load_safetensors;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

struct Quiet;

impl ProgressSink for Quiet {
    fn report(&self, _: f32, _: &str) {}
    fn cancelled(&self) -> bool {
        false
    }
}

fn weights(k: usize) -> Weights {
    let t = convert(&data(&format!("synth-unetpp-cv{k}.pt")), "mini", &Quiet).unwrap();
    Weights::load(&Params::new(t)).unwrap()
}

/// Worst difference relative to the reference's range.
fn worst(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let range = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    got.iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / range
}

#[test]
fn the_network_matches_konfai() {
    let io = load_safetensors(&data("synth-net-io.safetensors")).unwrap();
    let x = &io["input"];
    assert_eq!(x.shape, [1, 5, 64, 64]);
    for k in 0..2 {
        let w = weights(k);
        assert_eq!(w.in_ch, 5);
        let want = &io[&format!("logits_{k}")].data;
        let got = cpu::logits(&w, &x.data, 1, 64, 64);
        let e = worst(&got, want);
        assert!(e < 1e-4, "CV_{k} on the CPU: worst {e}");
        let tanh = &io[&format!("tanh_{k}")].data;
        let got = cpu::forward(&w, &x.data, 1, 64, 64);
        let e = worst(&got, tanh);
        assert!(e < 1e-4, "CV_{k} after tanh: worst {e}");
        // The same layers through burn (the GPU's code, on the CPU backend).
        let net = net::UnetPp::<burn::backend::NdArray>::upload(&w, &Default::default());
        let got = net.forward(&x.data, 1, 64, 64);
        let e = worst(&got, tanh);
        assert!(e < 1e-4, "CV_{k} through burn: worst {e}");
    }
}

/// The pipeline fixture.
struct Case {
    volume: Volume,
    body: Vec<u8>,
    mask: Vec<u8>,
    sct_tta0: Vec<i16>,
    sct_tta2: Vec<i16>,
}

fn case() -> Case {
    let bytes = std::fs::read(data("synth-pipeline.safetensors")).unwrap();
    let t = SafeTensors::deserialize(&bytes).unwrap();
    let i16s = |name: &str| -> Vec<i16> {
        let v = t.tensor(name).unwrap();
        assert_eq!(v.dtype(), Dtype::I16);
        v.data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| i16::from_le_bytes(c))
            .collect()
    };
    let u8s = |name: &str| -> Vec<u8> {
        let v = t.tensor(name).unwrap();
        assert_eq!(v.dtype(), Dtype::U8);
        v.data().to_vec()
    };
    let f64s = |name: &str| -> Vec<f64> {
        let v = t.tensor(name).unwrap();
        assert_eq!(v.dtype(), Dtype::F64);
        v.data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| f64::from_le_bytes(c))
            .collect()
    };
    let shape = t.tensor("scan").unwrap().shape().to_vec();
    let dims = [shape[2], shape[1], shape[0]];
    let spacing = f64s("spacing");
    let origin = f64s("origin");
    let data = i16s("scan");
    let volume = Volume {
        min_value: *data.iter().min().unwrap(),
        max_value: *data.iter().max().unwrap(),
        data,
        dims,
        spacing: [spacing[0], spacing[1], spacing[2]],
        origin: Vec3 {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        row_dir: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        col_dir: Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        normal: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        frame_of_reference_uid: String::new(),
    };
    Case {
        volume,
        body: u8s("body"),
        mask: u8s("mask"),
        sct_tta0: i16s("sct_tta0"),
        sct_tta2: i16s("sct_tta2"),
    }
}

#[test]
fn the_working_mask_matches_konfai() {
    let c = case();
    let o = Oriented::new(&c.volume, AxisOrder::Spl);
    let wdims = pre::working_dims(o.dims, o.spacing);
    assert_eq!(wdims, [8, 55, 72]);
    let m = pre::resample_nearest(&c.body, o.dims, o.spacing, wdims);
    let m = pre::dilate(&m, wdims, pre::DILATE);
    let differ = m.iter().zip(&c.mask).filter(|(a, b)| a != b).count();
    assert_eq!(
        differ,
        0,
        "{differ} of {} working-grid mask voxels differ",
        m.len()
    );
}

/// The fixture's checkpoints as a fine-tuned app, cached under a fresh
/// root.
fn app(tag: &str) -> (App, PathBuf) {
    let root = std::env::temp_dir().join(format!("rds_synth_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let custom: &'static CustomApp = Box::leak(Box::new(CustomApp {
        key: "mini",
        label: "TotalSynth miniature",
        subdir: "custom/mini",
        folder: data(""),
        checkpoints: vec![data("synth-unetpp-cv0.pt"), data("synth-unetpp-cv1.pt")],
    }));
    (App::Custom(custom), root)
}

fn compare(got: &[i16], want: &[i16], what: &str) {
    assert_eq!(got.len(), want.len());
    let mut differ = 0;
    let mut worst = 0;
    for (a, b) in got.iter().zip(want) {
        let d = (*a as i32 - *b as i32).abs();
        if d != 0 {
            differ += 1;
            worst = worst.max(d);
        }
    }
    // The network's float rounding (norms folded, another summation order)
    // differs from PyTorch's in the last bits, and that moves a voxel across
    // a half-precision step of the output (1 HU, 2 HU above 1024) now and
    // then: a few voxels in a thousand, never more than the step.
    assert!(
        differ * 200 <= got.len() && worst <= 2,
        "{what}: {differ} of {} voxels differ, worst {worst} HU",
        got.len()
    );
}

#[test]
fn one_checkpoint_matches_konfai() {
    let c = case();
    let (app, root) = app("t0");
    let opts = Options {
        checkpoints: vec![0],
        tta: 0,
        uncertainty: true,
        device: DevicePref::Cpu,
    };
    let r = run(
        &c.volume,
        Some(&c.body),
        app,
        &opts,
        &root,
        &Progress::default(),
    )
    .unwrap();
    assert_eq!(r.members, 1);
    assert_eq!(r.working_dims, [8, 55, 72]);
    compare(&r.hu, &c.sct_tta0, "CV_0");
    assert!(r.sd.unwrap().iter().all(|&v| v == 0.0));
    // A second run reads the converted cache.
    assert!(app.dir(&root).join(App::cache_name(0)).is_file());
    let again = run(
        &c.volume,
        Some(&c.body),
        app,
        &opts,
        &root,
        &Progress::default(),
    )
    .unwrap();
    assert_eq!(again.hu, r.hu);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn two_checkpoints_with_mirrors_match_konfai() {
    let c = case();
    let (app, root) = app("t2");
    let opts = Options {
        checkpoints: vec![0, 1],
        tta: 2,
        uncertainty: true,
        device: DevicePref::Cpu,
    };
    let r = run(
        &c.volume,
        Some(&c.body),
        app,
        &opts,
        &root,
        &Progress::default(),
    )
    .unwrap();
    assert_eq!(r.members, 6);
    compare(&r.hu, &c.sct_tta2, "CV_0 + CV_1, two mirrors");
    let sd = r.sd.unwrap();
    assert!(sd.iter().any(|&v| v > 0.0));
    let _ = std::fs::remove_dir_all(&root);
}

/// The full-size network (26.1 M parameters, the catalog graph with random
/// weights and calibrated norms, saved by KonfAI) on one real 512 x 512
/// stack: `RDS_SYNTH_FULL` names a folder with its `CV_0.pt` and
/// `full-io.safetensors` (input, logits), written outside the repository.
/// Also the CPU time of one pass.
#[test]
#[ignore]
fn the_full_size_network_matches_konfai() {
    let Some(dir) = std::env::var_os("RDS_SYNTH_FULL").map(PathBuf::from) else {
        eprintln!("RDS_SYNTH_FULL is not set");
        return;
    };
    let t = convert(&dir.join("CV_0.pt"), "full", &Quiet).unwrap();
    let w = Weights::load(&Params::new(t)).unwrap();
    let io = load_safetensors(&dir.join("full-io.safetensors")).unwrap();
    let x = &io["input"];
    assert_eq!(x.shape, [1, 5, 512, 512]);
    let start = std::time::Instant::now();
    let got = cpu::logits(&w, &x.data, 1, 512, 512);
    let secs = start.elapsed().as_secs_f64();
    let e = worst(&got, &io["logits"].data);
    eprintln!("one 512 x 512 pass on the CPU in {secs:.2} s; worst {e:e} of the range");
    assert!(e < 1e-4, "worst {e}");
    // Two stacks at once: the batch the CPU path runs.
    let mut two = x.data.clone();
    two.extend_from_slice(&x.data);
    let start = std::time::Instant::now();
    let both = cpu::logits(&w, &two, 2, 512, 512);
    eprintln!("a batch of two in {:.2} s", start.elapsed().as_secs_f64());
    assert_eq!(&both[..512 * 512], &got[..]);
}
