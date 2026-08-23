//! Promptable volumetric segmentation — a pure-Rust re-implementation of
//! [SegVol](https://github.com/BAAI-DCAI/SegVol) inference (Du et al.,
//! NeurIPS 2024, arXiv 2311.13385).
//!
//! Where [`crate::autoseg`] segments 117 fixed anatomical classes with no
//! interaction, this engine segments whatever it is pointed at: a 3-D box, a
//! set of clicks, or a structure name in plain text. That covers the things a
//! fixed-class model structurally cannot — lesions, targets, post-surgical
//! cavities — which is why the two engines coexist rather than compete.
//!
//! The network is a MONAI 3-D ViT image encoder feeding a SAM-style prompt
//! encoder and mask decoder, plus a frozen CLIP text tower for text prompts.
//! Its input shape is hard-locked to 32x256x256 (a learned 2048-token
//! position embedding and a shape-baked `LayerNorm` in the decoder both
//! enforce it), so inference is a static graph run twice: once over the whole
//! volume resized down to that shape, then again as a sliding window over a
//! crop around whatever the first pass found.
//!
//! Status: the image encoder ([`vit`]), prompt encoder ([`prompt`]) and mask
//! decoder ([`decoder`]) are assembled by [`net`]; [`preprocess`] and
//! [`infer`] carry a study through the two-pass pipeline; [`bpe`] and
//! [`clip`] turn a structure name into a text prompt. All of it is written
//! against the checkpoint layout recorded in [`layout`] and verified against
//! the real file. Still to come: the GPU backend and the interaction.

pub mod bpe;
pub mod clip;
pub mod config;
pub mod decoder;
pub mod infer;
pub mod layout;
pub mod net;
pub mod params;
pub mod preprocess;
pub mod prompt;
pub mod vit;
pub mod weights;
