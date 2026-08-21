//! Development probe: fetch the SegVol checkpoint and describe it.
//!
//! This is the first thing to run against the real weights. It proves the
//! checkpoint format is understood — that our pickle reader walks a
//! Hugging Face `state_dict()` and recovers every tensor's name, dtype and
//! shape — *before* any network code is written against it. If the parameter
//! arithmetic in `segvol::weights::EXPECTED_PARAMS` matches the file, the
//! architecture assumed by the plan is the architecture in the file.
//!
//! Only `data.pkl` is parsed, so everything after the download is instant;
//! the 724 MB of storage blobs are never read.
//!
//! ```text
//! cargo run --release --example segvol_probe -- [MODELS_DIR] [--keys] [--csv FILE]
//! ```
//!
//! `MODELS_DIR` defaults to `segvol_model/` next to the executable. `--keys`
//! lists every tensor; `--csv` writes name,dtype,shape,numel for offline
//! comparison against the reference implementation.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use rust_dicom_station::segvol::weights::{self, CHECKPOINT, EXPECTED_PARAMS};

struct Stderr;
impl rust_dicom_station::nn::cache::ProgressSink for Stderr {
    fn report(&self, _frac: f32, msg: &str) {
        // Carriage return keeps the download on one line.
        eprint!("\r\x1b[K{msg}");
        std::io::stderr().flush().ok();
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut models_dir: Option<PathBuf> = None;
    let mut list_keys = false;
    let mut csv: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--keys" => list_keys = true,
            "--csv" => csv = Some(PathBuf::from(args.next().expect("--csv needs a path"))),
            "-h" | "--help" => {
                eprintln!("usage: segvol_probe [MODELS_DIR] [--keys] [--csv FILE]");
                return Ok(());
            }
            other => models_dir = Some(PathBuf::from(other)),
        }
    }
    let models_dir = models_dir.unwrap_or_else(weights::default_models_dir);

    if !weights::is_cached(&CHECKPOINT, &models_dir) {
        eprintln!(
            "Fetching {} ({} MB) into {}",
            CHECKPOINT.name,
            CHECKPOINT.bytes / 1_000_000,
            models_dir.display()
        );
        eprintln!(
            "NOTE: the SegVol weights carry no license declaration. They are \
             downloaded to your machine at your request and are not \
             redistributed by this program."
        );
    }
    let path = weights::ensure_file(&CHECKPOINT, &models_dir, &Stderr)?;
    eprintln!("\rusing {}", path.display());

    let reader = weights::open_checkpoint(&path)?;

    // ---- inventory -------------------------------------------------------
    let mut total = 0usize;
    let mut dead = 0usize;
    let mut per_group: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    let mut noncontiguous = Vec::new();
    for (name, meta) in &reader.tensors {
        let n = meta.numel();
        total += n;
        if weights::is_dead_weight(name) {
            dead += n;
        }
        let e = per_group.entry(weights::group_of(name)).or_default();
        e.0 += 1;
        e.1 += n;
        *dtypes.entry(format!("{:?}", meta.dtype)).or_default() += n;
        if !meta.is_contiguous() {
            noncontiguous.push(name.clone());
        }
    }

    println!("tensors        {}", reader.tensors.len());
    println!("parameters     {total}");
    println!("expected       {EXPECTED_PARAMS}");
    println!();
    println!("{:<26} {:>7}  {:>13}", "group", "tensors", "parameters");
    for (g, (count, params)) in &per_group {
        println!("{g:<26} {count:>7}  {params:>13}");
    }
    println!();
    for (d, n) in &dtypes {
        println!("dtype {d:<10} {n:>13} values");
    }
    println!(
        "dead weights   {dead} ({} tensors, the 2-D mask_downscaling branch)",
        reader
            .tensors
            .iter()
            .filter(|(n, _)| weights::is_dead_weight(n))
            .count()
    );
    println!("live weights   {}", total - dead);

    // ---- the checks that matter -----------------------------------------
    let mut ok = true;
    if total != EXPECTED_PARAMS {
        println!(
            "\nMISMATCH: {total} parameters, expected {EXPECTED_PARAMS} \
             (difference {})",
            total as i64 - EXPECTED_PARAMS as i64
        );
        println!("The architecture in the file differs from the one the plan assumes.");
        ok = false;
    } else {
        println!("\nOK: parameter count matches the analytic total.");
    }
    if !noncontiguous.is_empty() {
        println!(
            "MISMATCH: {} non-contiguous tensors, e.g. {}",
            noncontiguous.len(),
            noncontiguous[0]
        );
        ok = false;
    }

    // A handful of tensors whose shapes pin down the parts of the
    // architecture that the published description gets wrong, and that the
    // port has to reproduce exactly.
    for (key, want) in [
        // learned absolute position embedding: 2048 tokens of dim 768, which
        // is what hard-locks the input to 32x256x256
        (
            "image_encoder.patch_embedding.position_embeddings",
            vec![1, 2048, 768],
        ),
        // patch embedding is a Linear over flattened (4,16,16) patches, not a
        // Conv3d: 4*16*16 = 1024 -> 768
        (
            "image_encoder.patch_embedding.patch_embeddings.1.weight",
            vec![768, 1024],
        ),
        // the random-Fourier prompt PE is a *buffer* and must be loaded, not
        // regenerated
        (
            "prompt_encoder.positional_encoding_gaussian_matrix",
            vec![3, 384],
        ),
        // the decoder LayerNorm normalizes over (C,D,H,W) jointly - 6.29 M
        // affine parameters and the second reason the shape is frozen
        (
            "mask_decoder.output_upscaling.1.weight",
            vec![192, 16, 32, 32],
        ),
        // CLIP ViT-B/32 text tower: vocab 49408, width 512
        (
            "text_encoder.clip_text_model.text_model.embeddings.token_embedding.weight",
            vec![49408, 512],
        ),
        // text is injected twice; this is the second path
        (
            "mask_decoder.txt_align_upscaled_embedding.weight",
            vec![96, 768],
        ),
    ] {
        match reader.tensors.iter().find(|(n, _)| n == key) {
            Some((_, m)) if m.shape == want => println!("OK: {key} {:?}", m.shape),
            Some((_, m)) => {
                println!("MISMATCH: {key} is {:?}, expected {want:?}", m.shape);
                ok = false;
            }
            None => {
                println!("MISSING: {key}");
                ok = false;
            }
        }
    }

    if list_keys {
        println!();
        for (name, meta) in &reader.tensors {
            println!("{name:<70} {:?} {:?}", meta.dtype, meta.shape);
        }
    }
    if let Some(p) = csv {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&p)?);
        writeln!(f, "name,dtype,shape,numel")?;
        for (name, meta) in &reader.tensors {
            let shape: Vec<String> = meta.shape.iter().map(|d| d.to_string()).collect();
            writeln!(
                f,
                "{name},{:?},{},{}",
                meta.dtype,
                shape.join(" "),
                meta.numel()
            )?;
        }
        eprintln!("wrote {}", p.display());
    }

    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
