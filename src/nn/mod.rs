//! Shared neural-network infrastructure.
//!
//! Nothing in here knows about a particular architecture. It is the part of
//! running a published PyTorch model natively that every engine needs, and
//! all three — [`autoseg`] (nnU-Net), [`segvol`] and [`medsam2`] — are built
//! on it and follow the same path: fetch the checkpoint once ([`cache`]),
//! parse the torch pickle ([`pickle`]), convert to `safetensors`, load from
//! that cache ever after, pick a device ([`device`]), and assemble the
//! network from a shape-checked view of the state dict ([`params`]). No
//! Python, no libtorch, no ONNX Runtime.
//!
//! Alongside the loading machinery are the dense CPU kernels a transformer
//! needs: [`tensor`] holds the two concrete shapes, [`linalg`] the matrix
//! multiply, normalizations and activations, and [`attention`] the one
//! attention routine every part of a network shares. Convolutions specific
//! to a U-Net remain in [`crate::autoseg::cpu`]; the MedSAM2 engine is
//! written against `burn` tensors instead and has its own small operator set
//! in [`crate::medsam2::ops`].
//!
//! [`autoseg`]: crate::autoseg
//! [`segvol`]: crate::segvol
//! [`medsam2`]: crate::medsam2

pub mod attention;
pub mod cache;
pub mod device;
pub mod half;
pub mod linalg;
pub mod params;
pub mod pickle;
pub mod tensor;
