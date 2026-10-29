//! rds-domain: everything of rust-dicom-station below the window - DICOM and
//! RT DICOM loading and writing, geometry and rendering primitives,
//! registration, structures, dose, 4D, the workflows, the archive and its
//! PACS server, the MCP server, the settings.
//!
//! A crate of its own for build time, as the engines are: an edit in the
//! viewer (`src/app/` of the root crate) recompiles the viewer and nothing
//! below it. The root crate re-exports every module here under the path it
//! had, so `rust_dicom_station::loader` and, inside the viewer,
//! `crate::loader` resolve as before; inside this crate, `crate::volume` and
//! `crate::autoseg` resolve through the re-exports of the two crates below.

pub mod anonymize;
pub mod archive;
pub mod audit;
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
pub mod iec;
pub mod imginfo;
pub mod livewire;
pub mod loader;
pub mod mesh3d;
pub mod models;
pub mod morphology;
pub mod motion;
pub mod pacs;
pub mod par;
pub mod portal;
pub mod projection;
pub mod propagate;
pub mod registration;
pub mod render;
pub mod report;
pub mod rooms;
pub mod rt_surface;
pub mod rtdose;
pub mod rtplan;
pub mod rtstruct;
pub mod sct;
pub mod segmentation;
pub mod sessions;
pub mod settings;
pub mod setup;
pub mod simulate;
pub mod sop;
pub mod structops;
pub mod templates;
pub mod testdata;
pub mod workflow;

#[cfg(feature = "mcp")]
pub mod mcp;

// The core types and the inference engines, under the paths they had when
// all of this was one crate.
pub use rds_core::{geometry, progress, volume};
pub use rds_engines::{
    autoseg, medsam2, nn, nninteractive, segresnet, segvol, synth, unet2d, vista3d, zoo,
};
