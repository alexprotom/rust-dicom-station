//! Promptable slice-propagating segmentation — a pure-Rust re-implementation
//! of [MedSAM2](https://github.com/bowang-lab/MedSAM2) inference (Ma et al.,
//! 2025, arXiv 2504.03600).
//!
//! Where [`crate::autoseg`] segments 117 fixed classes and
//! [`crate::segvol`] segments whatever it is pointed at inside a fixed
//! 32x256x256 box, this engine takes a **2-D prompt on one slice** — a box, a
//! click, or an existing mask — and propagates it through the stack at the
//! slice's own in-plane resolution. For CT, whose slices are natively
//! 512x512, that means no in-plane resampling at all.
//!
//! MedSAM2 is SAM 2.1 Hiera-Tiny fine-tuned on medical data with the input
//! halved to 512; the architecture is Meta's, unmodified. A volume is handed
//! to it the way SAM 2 is handed a video — slices are frames — so the port
//! needs SAM 2's memory bank as well as its image encoder: [`hiera`] and
//! [`neck`] encode a slice, [`prompt`] and [`decoder`] answer a prompt
//! against that encoding, and [`memory`] and [`memattn`] carry the answer to
//! the next slice.
//!
//! Everything runs through `burn`, which puts the whole graph on the GPU
//! (`wgpu`: Vulkan / DX12 / Metal, no CUDA toolkit) and falls back to a
//! pure-Rust CPU backend when there is no usable adapter — one implementation,
//! two backends, unlike the SegVol engine's split CPU/GPU pair.
//!
//! The port follows `docs/medsam2-plan.md` and is complete: checkpoint
//! layout and acquisition ([`layout`], [`weights`]), the image encoder
//! ([`hiera`], [`neck`]), the prompt encoder and mask decoder ([`prompt`],
//! [`decoder`], [`sam`]), the memory pair ([`memory`], [`memattn`]),
//! slice-to-slice propagation ([`track`], [`infer`]), preprocessing
//! ([`preprocess`], [`resample`]) and the one entry point the application
//! calls ([`engine`]). The user interface is `app::box_seg`.

pub mod config;
pub mod decoder;
pub mod engine;
pub mod hiera;
pub mod infer;
pub mod layers;
pub mod layout;
pub mod memattn;
pub mod memory;
pub mod model;
pub mod neck;
pub mod ops;
pub mod preprocess;
pub mod prompt;
pub mod resample;
pub mod sam;
pub mod track;
pub mod weights;
