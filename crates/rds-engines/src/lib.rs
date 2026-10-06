//! The native inference engines of rust-dicom-station: the nnU-Net family
//! ([`autoseg`]: TotalSegmentator's CT, MR and task models, MRSegmentator),
//! lungmask's 2-D U-Net ([`unet2d`]), MONAI's SegResNet family
//! ([`segresnet`]: the whole-body bundle, CT-FM), VISTA-3D ([`vista3d`]),
//! SegVol ([`segvol`]), MedSAM2 ([`medsam2`]) and nnInteractive
//! ([`nninteractive`]); the registry in front of
//! the automatic ones ([`zoo`]); and the neural-network plumbing they share
//! ([`nn`]): checkpoint download and conversion, device choice, tensors,
//! GEMM-backed linear algebra and attention. No Python, no libtorch, no
//! ONNX Runtime.
//!
//! A crate of its own for build time: with the `gpu` feature, burn's wgpu
//! backend is monomorphised wherever these engines are compiled, which is
//! more than half of the machine code the viewer used to build (and build
//! again for its unit tests). As a dependency it is compiled once and left
//! alone while the rest of the program is edited. The viewer re-exports the
//! modules under the paths they always had (`rust_dicom_station::nn`,
//! ...), so nothing that uses them changed.

pub mod autoseg;
pub mod medsam2;
pub mod nn;
pub mod nninteractive;
pub mod segresnet;
pub mod segvol;
pub mod unet2d;
pub mod vista3d;
pub mod zoo;

// The core types, under the crate-relative paths the engine code has always
// used (`crate::volume::Volume`, `crate::progress::ProgressSink`, ...).
pub(crate) use rds_core::{progress, volume};
// Only the unit tests build points in patient space.
#[cfg(test)]
pub(crate) use rds_core::geometry;
