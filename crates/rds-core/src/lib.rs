//! Core types of rust-dicom-station, shared by the viewer and its inference
//! engines: patient-space geometry ([`geometry`]), the voxel volume and its
//! lattice ([`volume`]), and the one progress handle every long operation
//! reports through ([`progress`]).
//!
//! They live in a crate of their own so the engines (`rds-engines`) can be
//! compiled apart from the viewer; the viewer re-exports all three under the
//! paths they always had (`rust_dicom_station::volume`, ...).

pub mod geometry;
pub mod progress;
pub mod volume;
