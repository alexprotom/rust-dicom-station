//! Generate the op-parity fixtures for the MedSAM2 port.
//!
//! Writes `tests/data/medsam2-ops.safetensors`: random inputs and, for every
//! primitive the engine implements, the output PyTorch, PIL or SAM 2 would
//! give, computed by the naive reference kernels in `tests/common/ops_ref.rs`.
//! Those kernels are proved against the copy of the file PyTorch itself wrote
//! by `tests/ops_fixtures.rs`, which is what makes a regenerated file stand
//! for the frameworks' semantics rather than for the engine's own.
//!
//! The committed file is the one PyTorch wrote and is what the kernels are
//! proved against, so the output path has to be given: write somewhere else
//! and copy it over only when the inventory itself has to change.
//!
//!     cargo run --example gen_ops_fixtures -- <out.safetensors> [--seed N]

#[path = "../tests/common/ops_ref.rs"]
mod ops_ref;

use std::path::PathBuf;

use rust_dicom_station::nn::cache::{save_tensor_map, StoreDtype};

fn main() -> anyhow::Result<()> {
    let mut path: Option<PathBuf> = None;
    let mut seed = 20260825u64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seed" => seed = args.next().expect("--seed N").parse()?,
            other => path = Some(PathBuf::from(other)),
        }
    }
    let Some(path) = path else {
        anyhow::bail!("usage: gen_ops_fixtures <out.safetensors> [--seed N]");
    };
    let tensors = ops_ref::generate(seed);
    save_tensor_map(&path, &tensors, StoreDtype::F32)?;
    println!(
        "wrote {}: {} tensors (seed {seed})",
        path.display(),
        tensors.len()
    );
    Ok(())
}
