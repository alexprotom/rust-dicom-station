//! Development probe: what a TotalSynth checkpoint holds, and the network's
//! output on a fixed input, to check against PyTorch.
//!
//! ```text
//! cargo run --release --example sct_probe -- <CV_k.pt> [--io <out.safetensors>] [--size N]
//! ```
//!
//! Prints the checkpoint's tensors (count, the naming they follow, a few
//! keys), whether they are a TotalSynth network, and the network read out
//! of them: input channels, stage widths and depths, decoder widths,
//! parameters. With `--io`, runs the network on the CPU on one stack of
//! `in_ch x N x N` (N = 512 by default; a smooth deterministic pattern in
//! [-1, 1]) and writes `input`, `logits` and `tanh` as safetensors, which
//! the same stack through KonfAI's model reproduces.

use std::collections::HashMap;
use std::path::PathBuf;

use rust_dicom_station::nn::cache::{save_tensor_map, StoreDtype, WTensor};
use rust_dicom_station::nn::params::Params;
use rust_dicom_station::nn::pickle::PthReader;
use rust_dicom_station::progress::Stderr;
use rust_dicom_station::synth::{self, layout};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or_else(|| {
        anyhow::anyhow!("usage: sct_probe <CV_k.pt> [--io <out.safetensors>] [--size N]")
    })?);
    let mut io: Option<PathBuf> = None;
    let mut size = 512usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--io" => io = args.next().map(PathBuf::from),
            "--size" => size = args.next().unwrap_or_default().parse()?,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }

    let r = PthReader::open(&path, "?Model.*")?;
    let keys: Vec<&str> = r.tensors.iter().map(|(k, _)| k.as_str()).collect();
    let naming = if keys.iter().any(|k| k.contains("decoder.blocks.")) {
        "segmentation_models_pytorch"
    } else if keys.iter().any(|k| k.contains("Stem.Conv_0")) {
        "KonfAI UNetPlusPlus.yml"
    } else if keys.iter().any(|k| k.contains("StemConv")) {
        "KonfAI parametric UNetPlusPlus"
    } else {
        "unknown"
    };
    println!("{}: {} tensors, {naming} names", path.display(), keys.len());
    for (k, m) in r.tensors.iter().take(6) {
        println!("  {k} {:?} {:?}", m.shape, m.dtype);
    }
    println!("  (and {} more)", keys.len().saturating_sub(6));
    println!(
        "a TotalSynth network: {}",
        layout::recognised(keys.iter().copied())
    );

    let t = synth::convert(&path, "probe", &Stderr)?;
    let w = layout::Weights::load(&Params::new(t))?;
    let params = |c: &layout::ConvW| c.w.len() + c.b.len();
    let mut n = params(&w.stem) + params(&w.head);
    for s in &w.stages {
        for b in s {
            n += params(&b.conv1) + params(&b.conv2) + b.skip.as_ref().map_or(0, params);
        }
    }
    for node in &w.nodes {
        n += params(&node.conv1) + params(&node.conv2);
    }
    println!(
        "in {} channels, stem {}, stages {:?} x {:?}",
        w.in_ch,
        w.stem.cout,
        w.stages.iter().map(|s| s[0].conv2.cout).collect::<Vec<_>>(),
        w.stages.iter().map(|s| s.len()).collect::<Vec<_>>()
    );
    println!(
        "decoder {}",
        layout::NODES
            .iter()
            .zip(&w.nodes)
            .map(|((d, l), node)| format!("x_{d}_{l} {}->{}", node.conv1.cin, node.conv2.cout))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("{n} parameters (batch norms folded)");

    if let Some(out) = io {
        let c = w.in_ch;
        let plane = size * size;
        let input: Vec<f32> = (0..c * plane)
            .map(|i| {
                let (ch, p) = (i / plane, i % plane);
                let (y, x) = ((p / size) as f32, (p % size) as f32);
                (0.03 * x + 0.05 * y + ch as f32).sin() * (0.02 * y).cos()
            })
            .collect();
        let t0 = std::time::Instant::now();
        let logits = synth::cpu::logits(&w, &input, 1, size, size);
        eprintln!("forward in {:.1} s", t0.elapsed().as_secs_f64());
        let tanh: Vec<f32> = logits.iter().map(|v| v.tanh()).collect();
        let mut map = HashMap::new();
        map.insert(
            "input".to_string(),
            WTensor {
                shape: vec![1, c, size, size],
                data: input,
            },
        );
        map.insert(
            "logits".to_string(),
            WTensor {
                shape: vec![1, 1, size, size],
                data: logits,
            },
        );
        map.insert(
            "tanh".to_string(),
            WTensor {
                shape: vec![1, 1, size, size],
                data: tanh,
            },
        );
        save_tensor_map(&out, &map, StoreDtype::F32)?;
        println!("written {}", out.display());
    }
    Ok(())
}
