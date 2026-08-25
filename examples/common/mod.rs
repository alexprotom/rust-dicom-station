//! What the headless examples share: argument parsing and the mask they all
//! end with. Cargo does not build `examples/common/mod.rs` as an example of
//! its own; each tool pulls it in with `mod common;`.

#![allow(dead_code)]

use std::path::Path;

use rust_dicom_station::volume::Volume;

/// A comma-separated list of numbers.
pub fn numbers(s: &str) -> Vec<f32> {
    s.split(',')
        .map(|x| x.trim().parse().expect("number"))
        .collect()
}

/// Report a mask's volume and optionally write it as raw `u8` bytes in
/// `Volume::data` order.
pub fn finish_mask(mask: &[u8], vol: &Volume, out: Option<&Path>) -> anyhow::Result<()> {
    let voxels = mask.iter().filter(|v| **v != 0).count();
    let cm3 = voxels as f64 * vol.spacing[0] * vol.spacing[1] * vol.spacing[2] / 1000.0;
    eprintln!("{voxels} voxels, {cm3:.1} cm3 on the original grid");
    if let Some(p) = out {
        std::fs::write(p, mask)?;
        eprintln!("wrote {} ({} bytes)", p.display(), mask.len());
    }
    Ok(())
}
