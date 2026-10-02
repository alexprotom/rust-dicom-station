//! The native inference engines of rust-dicom-station: TotalSegmentator's
//! nnU-Net ([`autoseg`]), SegVol ([`segvol`]) and MedSAM2 ([`medsam2`]), and
//! the neural-network plumbing all three share ([`nn`]): checkpoint download
//! and conversion, device choice, tensors, GEMM-backed linear algebra and
//! attention. No Python, no libtorch, no ONNX Runtime.
//!
//! A crate of its own for build time: with the `gpu` feature, burn's wgpu
//! backend is monomorphised wherever these engines are compiled, which is
//! more than half of the machine code the viewer used to build (and build
//! again for its unit tests). As a dependency it is compiled once and left
//! alone while the rest of the program is edited. The viewer re-exports the
//! four modules under the paths they always had (`rust_dicom_station::nn`,
//! ...), so nothing that uses them changed.

pub mod autoseg;
pub mod medsam2;
pub mod nn;
pub mod segvol;

// The core types, under the crate-relative paths the engine code has always
// used (`crate::volume::Volume`, `crate::progress::ProgressSink`, ...).
pub(crate) use rds_core::{progress, volume};
// Only the unit tests build points in patient space.
#[cfg(test)]
pub(crate) use rds_core::geometry;
