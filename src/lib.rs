//! rust-dicom-station library: the egui application, over the DICOM /
//! RT DICOM loading, geometry, registration and analysis of `rds-domain`
//! (re-exported here module by module).

pub mod app;
pub mod gfx_wgpu;
pub mod icon;

// Everything below the window is a crate of its own (crates/rds-domain), as
// the core types (crates/rds-core) and the inference engines
// (crates/rds-engines) are, so an edit in the viewer recompiles the viewer
// alone. Every module keeps the path it had here, `crate::loader` in the
// viewer and `rust_dicom_station::medsam2` outside alike.
pub use rds_domain::{
    anonymize, archive, audit, bodymask, contours, derived, dicom_export, dicomfile, dicomseg, drr,
    dvh, export, extras, fourd, gen_test_data, generate, gfx, iec, imginfo, livewire, loader,
    mesh3d, models, morphology, motion, pacs, par, portal, projection, propagate, registration,
    render, report, rooms, rt_surface, rtdose, rtplan, rtstruct, sct, segmentation, sessions,
    settings, setup, simulate, sop, structops, templates, testdata, workflow,
};
pub use rds_domain::{
    autoseg, medsam2, nn, nninteractive, segresnet, segvol, synth, unet2d, vista3d, zoo,
};
pub use rds_domain::{geometry, progress, volume};

#[cfg(feature = "mcp")]
pub use rds_domain::mcp;

/// The viewer's version, for the front ends whose own crate has another
/// (the Android and iOS packages): their log lines and panic files carry
/// this one.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
