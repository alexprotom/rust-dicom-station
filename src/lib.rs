//! rust-dicom-station library: DICOM / RT DICOM loading, geometry and
//! rendering primitives, plus the egui application.

pub mod anonymize;
pub mod app;
pub mod archive;
pub mod bodymask;
pub mod contours;
pub mod derived;
pub mod dicom_export;
pub mod dicomfile;
pub mod dicomseg;
pub mod drr;
pub mod dvh;
pub mod export;
pub mod extras;
pub mod fourd;
pub mod gen_test_data;
pub mod generate;
pub mod gfx;
pub mod icon;
pub mod imginfo;
pub mod livewire;
pub mod loader;
pub mod mesh3d;
pub mod models;
pub mod morphology;
pub mod motion;
pub mod par;
pub mod propagate;
pub mod registration;
pub mod render;
pub mod rt_surface;
pub mod rtdose;
pub mod rtplan;
pub mod rtstruct;
pub mod segmentation;
pub mod settings;
pub mod simulate;
pub mod structops;
pub mod templates;
pub mod testdata;
pub mod workflow;

#[cfg(feature = "mcp")]
pub mod mcp;

// The core types and the inference engines are crates of their own
// (crates/rds-core, crates/rds-engines), so the engines are compiled once
// rather than with every change to the viewer. Their modules keep the paths
// they had here, `crate::volume::Volume` and `rust_dicom_station::medsam2`
// alike.
pub use rds_core::{geometry, progress, volume};
pub use rds_engines::{autoseg, medsam2, nn, segvol};
