//! rust-dicom-station library: DICOM / RT DICOM loading, geometry and
//! rendering primitives, plus the egui application.

pub mod anonymize;
pub mod app;
pub mod autoseg;
pub mod dicom_export;
pub mod drr;
pub mod extras;
pub mod gen_test_data;
pub mod geometry;
pub mod loader;
pub mod medsam2;
pub mod mesh3d;
pub mod models;
pub mod nn;
pub mod progress;
pub mod propagate;
pub mod registration;
pub mod render;
pub mod rtdose;
pub mod rtplan;
pub mod rtstruct;
pub mod segmentation;
pub mod segvol;
pub mod settings;
pub mod simulate;
pub mod volume;
